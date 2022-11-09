import torch
from torch import nn

from deepinsight.core import (
    GetFuncNames,
    TorchTensorLazyInputDesc,
    TorchTensorListLazyInputDesc,
    TorchTensorConcreteInputDesc,
    TorchSizeConcreteInputDesc,
    TorchDtypeConcreteInputDesc,
    TorchDeviceConcreteInputDesc,
    PythonTypeConcreteInputDesc,
)


def register_func_call_convention(convention_type_name, cls_type):
    FuncCallConvention.convention_type_registry[convention_type_name] = cls_type


class FuncCallConvention(object):
    global_index = 0
    convention_type_registry = {}

    def __init__(
        self,
        module_name,
        func_name,
        func_signature,
        func_inp_args,
        func_inp_kwargs,
        index=None,
    ):
        """
        This class is to represent a function's calling conventions.

        Args:
            module_name (str): mod.__name__
            func_name (str): func.__name__
            func_signature (str): a string representing signature of the func.
            func_inp_args (list of InputDescBase): posional arguments for the func.
            func_inp_kwargs (dict, {str : InputDescBase}): keyword arguments for the func.
        """
        self.module_name = module_name
        self.func_name = func_name
        self.func_signature = func_signature
        self.func_inp_args = func_inp_args
        self.func_inp_kwargs = func_inp_kwargs
        self.index = FuncCallConvention.get_next_index() if index is None else index
        self.convention_type = None

    @property
    def module(self):
        raise NotImplementedError(
            "module property not implemented for FuncCallConvention"
        )

    @property
    def func(self):
        raise NotImplementedError(
            "func property not implemented for FuncCallConvention"
        )

    @classmethod
    def get_next_index(cls):
        cls.global_index += 1
        return str(cls.global_index)

    @staticmethod
    def from_yaml(cfg, index, yml_dir):
        return FuncCallConvention.convention_type_registry[cfg["type"]].from_yaml(
            cfg, index, yml_dir
        )

    def id(self):
        raise NotImplementedError(
            "id() function not implemented for FuncCallConvention"
        )

    def desc(self, brief=True):
        raise NotImplementedError(
            "desc() function not implemented for FuncCallConvention"
        )


supported_types = (tuple, list, str, bool, int, complex, type(None))
torch_nonfloat_dtypes = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)


def process_input(inp, export_data=False):
    is_torch_tensor = torch.is_tensor(inp)
    if is_torch_tensor:

        if export_data or inp.dtype in torch_nonfloat_dtypes:
            return TorchTensorConcreteInputDesc(inp)

        return TorchTensorLazyInputDesc(
            GetFuncNames.TORCH_RANDN_GET,
            list(inp.shape),
            inp.dtype,
            inp.requires_grad,
            inp.device,
        )

    elif isinstance(inp, torch.dtype):
        return TorchDtypeConcreteInputDesc(inp)
    elif isinstance(inp, torch.Size):
        return TorchSizeConcreteInputDesc(inp)
    elif isinstance(inp, torch.device):
        return TorchDeviceConcreteInputDesc(inp)
    elif isinstance(inp, float):
        return PythonTypeConcreteInputDesc(inp)
    else:
        is_tensor_list = False
        if isinstance(inp, (tuple, list)):
            is_tensor_list = all(torch.is_tensor(i) for i in inp)

        if is_tensor_list is True:
            return TorchTensorListLazyInputDesc(
                GetFuncNames.TORCH_LIST_RANDN_GET,
                [list(i.shape) for i in inp],
                [i.dtype for i in inp],
                [i.requires_grad for i in inp],
                [i.device for i in inp],
            )
        else:
            for typ in supported_types:
                if isinstance(inp, typ):
                    return PythonTypeConcreteInputDesc(inp)
    return type(inp).__name__


def process_inputs(op_name, inputs, export_data=False):
    if isinstance(inputs, tuple):
        inp_list = []
        for i, inp in enumerate(inputs):
            ret = process_input(inp, export_data)
            if isinstance(ret, str):
                raise TypeError(
                    f"For input {i} of `{op_name}`, type `{ret}` is UNSUPPORTED."
                )
            inp_list.append(ret)
        return tuple(inp_list)
    elif isinstance(inputs, dict):
        inp_dict = {}
        for key, val in inputs.items():
            ret = process_input(val, export_data)
            if isinstance(ret, str):
                raise TypeError(
                    f"For arg `{key}` of `{op_name}`, type `{ret}` is UNSUPPORTED."
                )
            inp_dict[key] = ret
        return inp_dict
    else:
        # Should not come to this branch
        print(f"WARNNING: Unexpected input type {type(inputs)}")
