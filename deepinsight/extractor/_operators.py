import shutil
import torch
import inspect as ins
import os
import yaml
import sys

from torch import nn
from ._shared import process_inputs, FuncCallConvention
from deepinsight.core import InputDescBase
from deepinsight.core._data import str_from_torch_device


str_to_torch_module_dict = {
    "torch": torch,
    "torch.Tensor": torch.Tensor,
    "Tensor": torch.Tensor,
    "F": torch.nn.functional,
    "nn.functional": torch.nn.functional,
    "torch.nn.functional": torch.nn.functional,
}


def patching_operator_hook(
    specify_ops=None, export_data=False, export_dir="/tmp/test/"
):
    if os.path.exists(export_dir):
        print(f"WARNNING: Overwriting existing folder {export_dir} in this run.")
        shutil.rmtree(export_dir)

    return _patching_hook_for_operators(
        specify_ops=specify_ops, export_data=export_data, export_dir=export_dir
    )


class OperatorCallConvention(FuncCallConvention):
    CONVENTION_TYPE = "operator"

    def __init__(
        self,
        module_name,
        func_name,
        func_signature,
        func_inp_args,
        func_inp_kwargs,
        index=None,
    ):
        super(OperatorCallConvention, self).__init__(
            module_name,
            func_name,
            func_signature,
            func_inp_args,
            func_inp_kwargs,
            index=index,
        )
        self.convention_type = OperatorCallConvention.CONVENTION_TYPE

    @property
    def module(self):
        return str_to_torch_module_dict[self.module_name]

    @property
    def func(self):
        return getattr(self.module, self.func_name)

    def desc(self, brief=True):
        ret_str = f"{self.module_name}.{self.func_name}"
        if brief is not True:
            ret_str += "("
            if self.func_inp_args:
                ret_str += ", ".join([str(arg) for arg in self.func_inp_args])

            if self.func_inp_kwargs:
                ret_str += "; " + ", ".join(
                    f"{key}={str(arg)}" for key, arg in self.func_inp_kwargs.items()
                )

            ret_str += ")"
            if self.func_signature != "":
                ret_str += f"[{self.func_signature}]"
        return ret_str

    def id(self):
        return f"{self.func_name}.{self.index}"

    @staticmethod
    def from_yaml(cfg, index, yml_dir):
        func_signature = cfg["func_signature"] if "func_signature" in cfg else ""

        func_inp_args = []
        func_inp_kwargs = {}
        for input_cfg in cfg["inputs"]["args"]:
            func_inp_args.append(InputDescBase.from_yaml(input_cfg, yml_dir))

        for k, input_cfg in cfg["inputs"]["kwargs"].items():
            func_inp_kwargs[k] = InputDescBase.from_yaml(input_cfg, yml_dir)

        return OperatorCallConvention(
            cfg["module_name"],
            cfg["func_name"],
            func_signature,
            func_inp_args,
            func_inp_kwargs,
            index=index,
        )

    def to_yaml(self, yml_dir):
        cfg = {}
        cfg["type"] = self.convention_type
        cfg["module_name"] = self.module_name
        cfg["func_name"] = self.func_name
        if self.func_signature != "":
            cfg["func_signature"] = self.func_signature

        input_configs = {}
        args = []
        for idx, input_arg in enumerate(self.func_inp_args):
            args.append(input_arg.to_yaml(yml_dir))

        kwargs = {}
        for k, input_arg in self.func_inp_kwargs.items():
            kwargs[k] = input_arg.to_yaml(yml_dir)

        input_configs["args"] = args
        input_configs["kwargs"] = kwargs

        cfg["inputs"] = input_configs

        print(self.index, cfg)
        return self.index, cfg


modules_to_patch = (torch, torch.Tensor, nn.functional)


def isfunc(mod, f):
    assert hasattr(mod, f)
    attr = getattr(mod, f)

    if f[0] == "_":
        return False  # Exclude functions starting with '_'
    if len(f) >= 2 and f[:2] == "__" and f[-2:] == "__":
        return False

    # Hardcode to ignore some functions/methods
    # Ignore functions to this list if they cause recursion
    ignore = ["size", "tolist", "dim", "is_storage", "item", "numpy"]
    # Also ignore the following functions related to tensor initialization
    ignore += ["zero_", "uniform_", "normal_", "fill_"]
    # and more
    ignore += [
        "copy_",
        "numel",
        "set_",
        "has_names",
        "index_select",
        "contiguous",
        "detach",
        "as_strided",
        "view_as",
        "cpu",
        "cuda",
        "bool",
        "float",
        "half",
        "double",
        "long",
        "to",
        "type",
    ]
    ignore += ["from_numpy", "tensor", "save", "stack", "load", "requires_grad_"]

    if f in ignore:
        return False

    ignore_patterns = ["storage", "stride", "has_torch_function", "new", "is_"]
    if any([s in f for s in ignore_patterns]):
        return False

    return ins.isroutine(attr)


def _patching_hook_for_operators(specify_ops, export_data, export_dir):
    if os.path.exists(export_dir):
        print(f"WARNNING: {export_dir} already exists, removing it now.")
        shutil.rmtree(export_dir)

    os.makedirs(export_dir, exist_ok=True)
    export_file_path = os.path.join(export_dir, "operator_input_info.yaml")

    def rewrite_func(mod, name):
        try:
            assert hasattr(mod, name)
            func = getattr(mod, name)

            # An example : torch.nn.functional.__name__
            # 'torch.nn.functional'
            func_name = func.__name__
            mod_name = mod.__name__
            try:
                signature = str(ins.signature(func))
            except:
                signature = ""
            if specify_ops is not None and func_name not in specify_ops.split(","):
                return

            def new_func(*args, **kwargs):
                with open(export_file_path, "a") as f:
                    try:
                        args_info = process_inputs(func_name, args, export_data)

                        if func_name == "empty":
                            print(kwargs["device"])
                            if "device" not in kwargs or kwargs["device"] is None:
                                tmp_tensor = torch.tensor([1, 2])
                                kwargs["device"] = str_from_torch_device(
                                    tmp_tensor.device
                                )
                            # if 'dtype' not in kwargs or kwargs['dtype'] is None:
                            #     tmp_tensor = torch.tensor([1,2])
                            #     kwargs['dtype'] = tmp_tensor.dtype

                        kwargs_info = process_inputs(func_name, kwargs, export_data)

                        fcc = OperatorCallConvention(
                            mod_name, func_name, signature, args_info, kwargs_info
                        )
                        idx, cfg = fcc.to_yaml(export_dir)
                        d = {}
                        d[idx] = cfg
                        yaml.dump(d, f)
                    except TypeError as e:
                        raise e
                        print(e)
                return func(*args, **kwargs)

            setattr(mod, name, new_func)
        except Exception as ex:
            print(f"`{name}` of `{mod}` NOT processed:", ex)

    for mod in modules_to_patch:
        for f in dir(mod):
            if isfunc(mod, f):
                rewrite_func(mod, f)
