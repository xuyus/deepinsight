import os
import torch
import inspect as ins
import pickle
import shutil
import yaml
from ._shared import process_inputs, FuncCallConvention
from deepinsight.core import InputDescBase
from deepinsight.core._data import generate_file_name


def patching_block_with_type_hook(
    specify_block_types=None, export_data=False, export_dir="/tmp/test/"
):
    if os.path.exists(export_dir):
        print(f"WARNNING: Overwriting existing folder {export_dir} in this run.")
        shutil.rmtree(export_dir)

    # For this case, specify_ops Must be a list containing the building block's type name.
    return _patching_hook_for_blocks(
        specify_block_types=specify_block_types,
        export_data=export_data,
        export_dir=export_dir,
    )


class BlockCallConvention(FuncCallConvention):
    CONVENTION_TYPE = "block"

    def __init__(
        self,
        module_instance: torch.nn.Module,
        module_name,
        func_name,
        func_signature,
        func_inp_args,
        func_inp_kwargs,
        index=None,
    ):
        super(BlockCallConvention, self).__init__(
            module_name,
            func_name,
            func_signature,
            func_inp_args,
            func_inp_kwargs,
            index=index,
        )
        cls_type = type(module_instance)
        self.module_type = str(cls_type.__module__ + "." + cls_type.__qualname__)
        self.module_instance = module_instance
        self.convention_type = BlockCallConvention.CONVENTION_TYPE

    @property
    def module(self):
        return self.module_instance

    @property
    def func(self):
        return self.module_instance.__call__

    def desc(self, brief=True):
        ret_str = f"{self.module_name}({self.module_type})->{self.func_name}"
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
        return f"{self.module_name}.{self.func_name}.{self.index}"

    @staticmethod
    def from_yaml(cfg, index, yml_dir):
        func_signature = cfg["func_signature"] if "func_signature" in cfg else ""
        func_inp_args = []
        func_inp_kwargs = {}
        for input_cfg in cfg["inputs"]["args"]:
            func_inp_args.append(InputDescBase.from_yaml(input_cfg, yml_dir))

        for k, input_cfg in cfg["inputs"]["kwargs"].items():
            func_inp_kwargs[k] = InputDescBase.from_yaml(input_cfg, yml_dir)

        module_name = cfg["module_name"]
        file_path = os.path.join(yml_dir, cfg["module_from"])
        with open(file_path, "rb") as f:
            module_instance = pickle.load(f)
        cls_type = type(module_instance)
        module_type = str(cls_type.__module__ + "." + cls_type.__qualname__)
        assert module_type == cfg["module_type"]

        return BlockCallConvention(
            module_instance,
            module_name,
            cfg["func_name"],
            func_signature,
            func_inp_args,
            func_inp_kwargs,
            index=index,
        )

    def to_yaml(self, yml_dir):
        file_name = "{}.pkl".format(generate_file_name())
        file_path = os.path.join(yml_dir, file_name)
        with open(file_path, "ab") as f:
            pickle.dump(self.module_instance, f)

        cfg = {}
        cfg["type"] = self.convention_type
        cfg["module_from"] = file_path
        cfg["module_type"] = self.module_type
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


def _rewrite_module_cls_forward_func(mod, name, export_data, export_dir):
    export_file_path = os.path.join(export_dir, "block_input_info.yaml")

    try:
        assert name == "forward"
        assert hasattr(mod, name)
        func = getattr(mod, name)

        def fullname(klass):
            module = klass.__module__
            if module == "builtins":
                return klass.__qualname__  # avoid outputs like 'builtins.str'
            return module + "." + klass.__qualname__

        mod_name = fullname(mod)
        func_name = func.__name__
        try:
            signature = str(ins.signature(func))
        except:
            signature = ""

        def new_func(self, *args, **kwargs):
            with open(export_file_path, "a") as f:
                try:
                    args_info = process_inputs(func_name, args, export_data)
                    kwargs_info = process_inputs(func_name, kwargs, export_data)

                    fcc = BlockCallConvention(
                        self,
                        mod_name,
                        func_name,
                        signature,
                        args_info,
                        kwargs_info,
                        index=None,
                    )
                    idx, cfg = fcc.to_yaml(export_dir)
                    d = {}
                    d[idx] = cfg
                    yaml.dump(d, f)
                except TypeError as e:
                    print(e)
            return func(self, *args, **kwargs)

        setattr(mod, name, new_func)
        # Be noted: we cannot use following approach, which only modify per-instance forward function.
        # import types
        # mod.forward = types.MethodType(new_func, mod)
    except Exception as ex:
        print(f"`{name}` of `{mod}` NOT processed:", ex)


def _patching_hook_for_blocks(specify_block_types, export_data, export_dir):
    if os.path.exists(export_dir):
        print(f"WARNNING: {export_dir} already exists, removing it now.")
        shutil.rmtree(export_dir)

    os.makedirs(export_dir, exist_ok=True)

    for mod in specify_block_types:
        for f in dir(mod):
            if f == "forward":
                print("hooking nn.Module forward path for ", mod)
                _rewrite_module_cls_forward_func(mod, f, export_data, export_dir)
