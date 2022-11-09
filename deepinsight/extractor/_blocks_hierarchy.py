from deepinsight.extractor._operators import (
    isfunc,
    modules_to_patch,
    OperatorCallConvention,
)
import os
import pickle
import torch
import copy
import inspect as ins
import pickle
import os
import shutil
from collections import OrderedDict
from ._shared import process_inputs, FuncCallConvention
from deepinsight.core import InputDescBase
from deepinsight.core._data import generate_file_name, str_from_torch_device
from deepinsight.extractor._blocks import BlockCallConvention
import yaml


def patching_block_and_operator_with_hierarchy_hook(
    model, export_data=False, export_dir="/tmp/test/"
):
    if os.path.exists(export_dir):
        print(f"WARNNING: Overwriting existing folder {export_dir} in this run.")
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    return _patching_hook_blocks_and_operators_with_hierarchy(
        model, export_data=export_data, export_dir=export_dir
    )


class ModuleInstanceWrapper(object):
    def __init__(self, nn_module_intance, idx, indent):
        self.nn_module = nn_module_intance
        self.idx = idx
        self.indent = indent


class BlockHierarchyCallConvention(FuncCallConvention):
    CONVENTION_TYPE = "block_hierarchy"

    def __init__(
        self,
        module_name,
        func_name,
        func_signature,
        func_inp_args,
        func_inp_kwargs,
        index=None,
    ):
        super(BlockHierarchyCallConvention, self).__init__(
            module_name,
            func_name,
            func_signature,
            func_inp_args,
            func_inp_kwargs,
            index=index,
        )
        self.convention_type = BlockHierarchyCallConvention.CONVENTION_TYPE

        self.module_type = None
        self.module_instance = None

    @property
    def module(self):
        assert self.module_instance is not None
        return self.module_instance.nn_module

    @property
    def func(self):
        assert self.module_instance is not None
        return self.module_instance.nn_module.__call__

    def set_module_instance(self, module_instance):
        assert isinstance(module_instance, ModuleInstanceWrapper)
        cls_type = type(module_instance.nn_module)
        self.module_type = str(cls_type.__module__ + "." + cls_type.__qualname__)
        self.module_instance = module_instance
        return self

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
        m_path = os.path.join(yml_dir, "serialized_nn_module", str(index))
        try:
            with open(m_path, "rb") as f:
                module_instance = pickle.load(f)
        except Exception as e:
            print("Fail to load serilized nn module for idx: {}".format(index))
            raise e

        func_signature = cfg["func_signature"] if "func_signature" in cfg else ""
        func_inp_args = []
        func_inp_kwargs = {}
        for input_cfg in cfg["inputs"]["args"]:
            func_inp_args.append(InputDescBase.from_yaml(input_cfg, yml_dir))

        for k, input_cfg in cfg["inputs"]["kwargs"].items():
            func_inp_kwargs[k] = InputDescBase.from_yaml(input_cfg, yml_dir)

        module_name = cfg["module_name"]
        return BlockHierarchyCallConvention(
            module_name,
            cfg["func_name"],
            func_signature,
            func_inp_args,
            func_inp_kwargs,
            index=index,
        ).set_module_instance(module_instance)

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


reversed_module = {}


def _patching_hook_blocks_and_operators_with_hierarchy(model, export_data, export_dir):
    global reversed_module
    for mod_name in dir(torch.nn):
        m = getattr(torch.nn, mod_name)
        if ins.isclass(m) and issubclass(m, torch.nn.Module):
            print("hooking nn.Module class for ", mod_name)
            _rewrite_module_cls_forward_func(
                m, export_data, export_dir, reversed_module
            )

    _patching_hook_for_blocks_with_hierarchy(model, export_data, export_dir)

    _patching_hook_for_operators_hierarchy(export_dir)


def _patching_hook_for_blocks_with_hierarchy(model, export_data, export_dir):
    global reversed_module
    module_idx_dict = OrderedDict()
    scanned_modules = OrderedDict()
    export_model_info_dir = os.path.join(export_dir, "serialized_nn_module")

    if os.path.exists(export_model_info_dir):
        print(f"WARNNING: {export_model_info_dir} already exists, removing it now.")
        shutil.rmtree(export_model_info_dir)
    os.makedirs(export_model_info_dir, exist_ok=True)

    model_name = "model"

    def save_nn_modules_recursively(model, name, prefix_idx, intent=0):
        print("\t" * intent, prefix_idx, name, type(model))
        if model in scanned_modules:
            prefix_idx = scanned_modules[model]
        else:
            scanned_modules[model] = prefix_idx
        module_idx_dict[prefix_idx] = [name, model]
        try:
            with open(os.path.join(export_model_info_dir, prefix_idx), "ab") as f:
                with torch.no_grad():
                    pickle.dump(
                        ModuleInstanceWrapper(
                            copy.deepcopy(model).cpu(), prefix_idx, intent
                        ),
                        f,
                    )
        except Exception as e:
            print(e)

        local_idx = 0
        for name, child in model.named_children():
            new_prefix_idx = prefix_idx + "." + str(local_idx)
            save_nn_modules_recursively(child, name, new_prefix_idx, intent + 1)
            local_idx += 1

    save_nn_modules_recursively(model, model_name, "root", 0)

    for k, name_child_pair in module_idx_dict.items():
        _rewrite_module_instance_forward_func(
            name_child_pair[1], name_child_pair[0], False, export_dir, k
        )
        reversed_module[name_child_pair[1]] = [k, 0]


def _rewrite_module_instance_forward_func(
    module_instance, module_name, export_data, export_dir, idx
):
    index = idx
    export_input_info_yml = os.path.join(export_dir, "block_input_info.yaml")
    try:
        func_name = "forward"
        assert hasattr(module_instance, func_name)
        func = getattr(module_instance, func_name)
        try:
            signature = str(ins.signature(func))
        except:
            signature = ""

        def new_func(self, *args, **kwargs):
            with open(export_input_info_yml, "a") as f:
                try:
                    args_info = process_inputs(func_name, args, export_data)
                    kwargs_info = process_inputs(func_name, kwargs, export_data)

                    fcc = BlockHierarchyCallConvention(
                        module_name,
                        func_name,
                        signature,
                        args_info,
                        kwargs_info,
                        index=index,
                    )
                    idx, cfg = fcc.to_yaml(export_dir)
                    d = {}
                    d[idx] = cfg
                    yaml.dump(d, f)
                except TypeError as e:
                    print(e)
            return func(*args, **kwargs)

        # setattr(mod, name, new_func)
        # Be noted: we use following approach to modify per-instance forward function.
        import types

        module_instance.forward = types.MethodType(new_func, module_instance)
    except Exception as ex:
        print(f"`{func_name}` of `{module_instance}` NOT processed:", ex)


def _get_calling_function(fr, depth=0):
    """finds the calling function in many decent cases."""
    co = fr.f_code
    func = None
    try:
        func = getattr(fr.f_locals["self"], co.co_name)
    except Exception as e:  # (KeyError, AttributeError, Exception):
        # print("ddddddddddddddd", e)
        pass

    if func:
        return func
    else:
        if depth > 10:
            raise AttributeError("func not found")
        return _get_calling_function(fr.f_back, depth=depth + 1)


def _get_calling_module_instance(fr):
    func = _get_calling_function(fr)
    if func is not None:
        return func.__self__

    return None


def _get_calling_tracked_module_instance(fr, reversed_module, depth=0):
    """finds the calling function in many decent cases."""
    co = fr.f_code
    func = None
    try:
        func = getattr(fr.f_locals["self"], co.co_name)
    except Exception as e:  # (KeyError, AttributeError, Exception):
        # print("ddddddddddddddd", e)
        pass

    if func and func.__self__ in reversed_module:
        return func.__self__
    else:
        if depth > 10:
            raise AttributeError("func not found")
        return _get_calling_tracked_module_instance(
            fr.f_back, reversed_module, depth=depth + 1
        )
    return None


def _rewrite_module_cls_forward_func(mod, export_data, export_dir, reversed_module):
    export_file_path = os.path.join(export_dir, "block_input_info_part2.yaml")

    try:
        name = "forward"
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
            if self not in reversed_module:
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!find module not in reversed_module",
                    mod_name,
                )
                with open(export_file_path, "a") as f:
                    try:
                        previous_frame = ins.currentframe().f_back
                        module_instance = _get_calling_tracked_module_instance(
                            previous_frame, reversed_module
                        )
                        if module_instance is None:
                            print(
                                "did not find the module instance for {}".format(
                                    mod_name
                                )
                            )
                            return func(self, *args, **kwargs)

                        if module_instance not in reversed_module:
                            print(
                                "did not find the module instance for {} in reversed_module".format(
                                    mod_name
                                )
                            )
                            return func(self, *args, **kwargs)

                        if module_instance:
                            module_idx, next_index = reversed_module[module_instance]
                            next_idx = "{}.f{}".format(module_idx, next_index)
                            reversed_module[module_instance][1] += 1

                        args_info = process_inputs(func_name, args, export_data)
                        kwargs_info = process_inputs(func_name, kwargs, export_data)

                        # fcc = BlockHierarchyCallConvention(mod_name, func_name, signature, args_info, kwargs_info, index=next_idx)
                        fcc = BlockCallConvention(
                            self,
                            mod_name,
                            func_name,
                            signature,
                            args_info,
                            kwargs_info,
                            index=next_idx,
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


functional_parent_id_dict = {}


def _patching_hook_for_operators_hierarchy(export_dir):
    global reversed_module
    # if os.path.exists(export_dir):
    #     print(f'WARNNING: {export_dir} already exists, removing it now.')
    #     shutil.rmtree(export_dir)
    export_data = False
    os.makedirs(export_dir, exist_ok=True)
    export_file_path = os.path.join(export_dir, "operator_input_info.yaml")

    def rewrite_func(mod, name):
        try:
            assert hasattr(mod, name)
            func = getattr(mod, name)

            # An example : torch.nn.functional.__name__
            # 'torch.nn.functional'
            mod_name = mod.__name__
            func_name = func.__name__

            try:
                signature = str(ins.signature(func))
            except:
                signature = ""

            def new_func(*args, **kwargs):
                try:
                    previous_frame = ins.currentframe().f_back
                    (
                        filename,
                        line_number,
                        function_name,
                        lines,
                        index,
                    ) = ins.getframeinfo(previous_frame)
                    print(func_name, filename, line_number, function_name, lines, index)

                    if filename.startswith(
                        "/opt/conda/lib/python3.7/site-packages/torch/nn/"
                    ):
                        return func(*args, **kwargs)

                    co = previous_frame.f_code
                    module_instance = _get_calling_module_instance(previous_frame)
                    if module_instance is None:
                        print(
                            "did not find the module instance for {}".format(
                                function_name
                            )
                        )
                        return func(*args, **kwargs)

                    if module_instance not in reversed_module:
                        print(
                            "did not find the module instance for {} in reversed_module".format(
                                function_name
                            )
                        )
                        return func(*args, **kwargs)

                    if module_instance:
                        module_idx, next_index = reversed_module[module_instance]
                        next_idx = "{}.f{}".format(module_idx, next_index)
                        reversed_module[module_instance][1] += 1
                        functional_parent_id_dict[
                            ins.currentframe().f_code
                        ] = module_instance

                    with open(export_file_path, "a") as f:
                        try:
                            args_info = process_inputs(func_name, args, export_data)

                            # empty did not has tensor inputs, device is the param to know where to run the op.
                            # The other hand, ORTModule ONLY get the device from the tensor input.
                            # if func_name == 'empty':
                            #     if 'device' not in kwargs or kwargs['device'] is None:
                            #         tmp_tensor = torch.tensor([1, 2])
                            #         kwargs['device'] = str_from_torch_device(
                            #             tmp_tensor.device)

                            kwargs_info = process_inputs(func_name, kwargs, export_data)

                            fcc = OperatorCallConvention(
                                mod_name,
                                func_name,
                                signature,
                                args_info,
                                kwargs_info,
                                index=next_idx,
                            )
                            idx, cfg = fcc.to_yaml(export_dir)
                            d = {}
                            d[idx] = cfg
                            yaml.dump(d, f)
                        except TypeError as e:
                            raise e
                            print(e)
                except Exception as ex:
                    print("fail to export the op: {}".format(ex))
                return func(*args, **kwargs)

            setattr(mod, name, new_func)
        except Exception as ex:
            print(f"`{name}` of `{mod}` NOT processed:", ex)

    for mod in modules_to_patch:
        for f in dir(mod):

            if isfunc(mod, f):
                rewrite_func(mod, f)
