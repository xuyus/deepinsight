import csv
import os
import pickle

import numpy as np
import torch
import copy
import string
import random
from collections import OrderedDict


def npyarray2string(npy_arr):
    return str(npy_arr.shape) + str(npy_arr.dtype)


def torch_tensor2string(tensor):
    return str(tensor.shape) + str(tensor.dtype)


def shape_from_str(int_list_str):
    if not isinstance(int_list_str, str):
        raise ValueError(
            "list_of_int type not expected {}, value: {}".format(
                type(int_list_str), int_list_str
            )
        )
    assert int_list_str[0] == "[" and int_list_str[-1] == "]"
    int_list_str = int_list_str[1:-1]  # ignore the final ')'
    int_list = int_list_str.split(",")
    if len(int_list) > 0:
        return [int(s) for s in int_list]
    else:
        return []


def generate_file_name():
    S = 10  # number of characters in the string.
    # call random.choices() string module to find the string in Uppercase + numeric data.
    ran = "".join(random.choices(string.ascii_uppercase + string.digits, k=S))
    return ran


def str_from_torch_device(device):
    if device.index is None:
        return "{}".format(device.type)
    else:
        return "{}:{}".format(device.type, device.index)


def str_to_torch_device(device_str):
    return torch.device(device_str)


# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

torch_to_numpy_dtype_dict = dict((y, x) for x, y in numpy_to_torch_dtype_dict.items())


str_to_torch_dtype_dict = {
    "torch.bool": torch.bool,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.complex64": torch.complex64,
    "torch.complex128": torch.complex128,
}

torch_dtype_to_str_dict = dict((y, x) for x, y in str_to_torch_dtype_dict.items())

str_to_numpy_dtype_dict = {
    "np.bool": np.bool,
    "np.uint8": np.uint8,
    "np.int8": np.int8,
    "np.int16": np.int16,
    "np.int32": np.int32,
    "np.int64": np.int64,
    "np.float16": np.float16,
    "np.float32": np.float32,
    "np.float64": np.float64,
    "np.complex64": np.complex64,
    "np.complex128": np.complex128,
}

numpy_dtype_to_str_dict = dict((y, x) for x, y in str_to_numpy_dtype_dict.items())


class InputDescBase:
    def __init__(self):
        self._data = None

    def get_data(self):
        if self._data is None:
            self._data = self._create_value()

        return self._data

    @staticmethod
    def from_yaml(input_config, yml_dir):
        get_func = input_config["get_func"]
        if get_func in GetFuncName_to_ConcreteInput_Mappings:
            return GetFuncName_to_ConcreteInput_Mappings[get_func].from_yaml(
                input_config, yml_dir
            )

        if get_func in GetFuncName_to_LazyInputDesc_Mappings:
            input_desc_type, _ = GetFuncName_to_LazyInputDesc_Mappings[get_func]
            return input_desc_type.from_yaml(input_config, yml_dir, get_func)
        raise RuntimeError("Fail to find the get_func : {}".format(get_func))


#### Concrete Input Desc Section STARTS ####


class NumpyConcreteInputDesc(InputDescBase):
    """
    This class is used to depict an fixed input.

    """

    def __init__(self, input_value):
        """
        Args:
            input_values (list of torch tensor or numpy value or python list):
                The inputs that can be directly used by training without any post-processing.
        """
        super(NumpyConcreteInputDesc, self).__init__()
        self.value = input_value
        self.shape = [] if input_value.ndim == 0 else list(input_value.shape)
        if isinstance(self.value, np.ndarray):
            self.dtype = self.value.dtype
        else:
            raise RuntimeError(
                "NumpyConcreteInputDesc cannot represent a non-numpy value."
            )

    def _create_value(self):
        return self.value

    def __str__(self):
        return "NP{{{}}}".format(str(self.shape))

    def to_yaml(self, yml_dir):
        input_config = {}
        tensor_file_name = generate_file_name() + ".npy"
        filename = os.path.join(yml_dir, tensor_file_name)
        # filename = os.path.join(yml_dir, self._generate_name())
        self.save_as(filename)
        np.save(filename, self.value)
        input_config["get_func"] = GetFuncName_to_ConcreteInput_Reverse_Mappings[
            NumpyConcreteInputDesc
        ]
        input_config["value_from"] = filename
        input_config["shape"] = str(self.shape)
        input_config["dtype"] = numpy_dtype_to_str_dict[self.dtype]
        return input_config

    @staticmethod
    def from_yaml(input_config, yml_dir):
        value = np.load(os.path.join(yml_dir, input_config["value_from"]))
        dtype = str_to_numpy_dtype_dict[input_config["dtype"]]
        assert dtype == value.dtype
        shape = [] if value.ndim == 0 else list(value.shape)
        assert shape_from_str(input_config["shape"]) == shape
        return NumpyConcreteInputDesc(value)


class PythonTypeConcreteInputDesc(InputDescBase):
    """
    This class is used to depict an fixed input.

    """

    def __init__(self, input_value):
        """
        Args:
            input_values (list of torch tensor or numpy value or python list):
                The inputs that can be directly used by training without any post-processing.
        """
        super(PythonTypeConcreteInputDesc, self).__init__()
        self.value = input_value
        # self.dtype = input_type

    def __str__(self):
        return "PT{{{}}}".format(self.value)

    def _create_value(self):
        return self.value

    def to_yaml(self, yml_dir):
        input_config = {}
        input_config["get_func"] = GetFuncName_to_ConcreteInput_Reverse_Mappings[
            PythonTypeConcreteInputDesc
        ]
        # input_config['dtype'] = self.dtype
        input_config["value_from"] = self.value
        return input_config

    @staticmethod
    def from_yaml(input_config, yml_dir):
        return PythonTypeConcreteInputDesc(input_config["value_from"])


#### Concrete Input Desc Section ENDS ####

#### Lazy Input Desc Section STARTS ####


def np_rand(shape, dtype):
    # add a np.array(*) in case the shape is empty and rand() returns a float.
    return np.array(np.random.rand(*shape)).astype(dtype)


def np_rand_bi_p0_5(shape, dtype):
    p = 0.5
    return np.random.binomial(1, p, shape).astype(dtype)


class LazyInputDescBase(InputDescBase):
    """
    This class is used to depict an input that can be allocated/randomized before the benchmark started.

    """

    def __init__(self, get_func_name, shape, dtype):
        super(LazyInputDescBase, self).__init__()
        assert isinstance(shape, (list,))
        self._shape = shape
        self._dtype = dtype
        self._get_func_name = get_func_name


class LazyListInputDescBase(InputDescBase):
    """
    This class is used to depict an input that can be allocated/randomized before the benchmark started.

    """

    def __init__(self, get_func_name, shapes, dtypes):
        super(LazyListInputDescBase, self).__init__()
        assert isinstance(shapes, (list,))
        self._shapes = shapes
        self._dtypes = dtypes
        self._get_func_name = get_func_name


class NumpyLazyInputDesc(LazyInputDescBase):
    """
    This class is used to depict an input that can be allocated/randomized before the benchmark started.

    """

    def __init__(self, generator_func_name, shape, dtype):
        super(NumpyLazyInputDesc, self).__init__(generator_func_name, shape, dtype)
        assert isinstance(dtype, np.dtype)

    def _create_value(self):
        _, generator_func = GetFuncName_to_LazyInputDesc_Mappings[self._get_func_name]
        return generator_func(self._shape, self._dtype)

    def __str__(self):
        return "NP{{{}}}".format(str(self._shape))

    def to_yaml(self, yml_dir):
        input_config = {}
        input_config["get_func"] = self._get_func_name
        input_config["value_from"] = str(self._shape)
        input_config["dtype"] = numpy_dtype_to_str_dict[self._dtype]
        return input_config

    @staticmethod
    def from_yaml(input_config, yml_dir, generator_func_name):
        return NumpyLazyInputDesc(
            generator_func_name,
            shape_from_str(input_config["value_from"]),
            str_to_numpy_dtype_dict[input_config["dtype"]],
        )


# class LazyBinaryInputDesc(LazyInputDescBase):
#     def __init__(self, input_shape, dtype, as_torch_tensor=False, requires_grad=False):
#         super(LazyBinaryInputDesc, self).__init__(input_shape, dtype, as_torch_tensor=as_torch_tensor,
#                                                   requires_grad=requires_grad)

#     def _create_value(self, p=0.5):
#         data = np.random.binomial(1, p, self.input_shape).astype(self.dtype)

#         return data


# with torch.no_grad():
#     self.value = tensor_input.detach().cpu()


#### Lazy Input Desc Section STARTS ####
#### Torch Input Desc Section STARTS ####


class TorchTensorConcreteInputDesc(InputDescBase):
    """
    This class is used to depict an fixed input.

    """

    def __init__(self, tensor_input):
        """
        Args:
            input_values (list of torch tensor or numpy value or python list):
                The inputs that can be directly used by training without any post-processing.
        """
        super(TorchTensorConcreteInputDesc, self).__init__()
        with torch.no_grad():
            self.device = tensor_input.detach().device
            self.value = tensor_input.detach().cpu().to(self.device)
            self.requires_grad = tensor_input.detach().requires_grad
            self.shape = [] if self.value.dim() == 0 else list(self.value.shape)
            self.dtype = tensor_input.detach().dtype

    def _create_value(self):
        with torch.no_grad():
            return self.value.to(dtype=self.dtype, device=self.device).requires_grad_(
                requires_grad=self.requires_grad
            )

    def __str__(self):
        return "TS{{{}}}".format(str(self.shape))

    def to_yaml(self, yml_dir):
        # input: ramdom_tensor([8, 512, 1024], float, required_grad=True)
        input_config = {}
        tensor_file_name = generate_file_name() + ".pt"
        filename = os.path.join(yml_dir, tensor_file_name)
        torch.save(self.value, filename)
        input_config["get_func"] = GetFuncName_to_ConcreteInput_Reverse_Mappings[
            TorchTensorConcreteInputDesc
        ]
        input_config["value_from"] = tensor_file_name
        input_config["shape"] = str(self.shape)
        input_config["dtype"] = torch_dtype_to_str_dict[self.dtype]
        input_config["requires_grad"] = self.requires_grad
        input_config["device"] = str_from_torch_device(self.device)
        return input_config

    @staticmethod
    def from_yaml(input_config, yml_dir):
        value = torch.load(os.path.join(yml_dir, input_config["value_from"]))
        assert str_to_torch_dtype_dict[input_config["dtype"]] == value.dtype
        shape = [] if value.dim() == 0 else list(value.shape)
        assert shape_from_str(input_config["shape"]) == shape
        value.requires_grad_(input_config["requires_grad"])
        assert input_config["device"] == str_from_torch_device(value.device)
        return TorchTensorConcreteInputDesc(value)


class TorchDtypeConcreteInputDesc(InputDescBase):
    """
    This class is used to depict an fixed input of type torch.dtype.

    """

    def __init__(self, input_value):
        """
        Args:
            input_values (list of torch tensor or numpy value or python list):
                The inputs that can be directly used by training without any post-processing.
        """
        super(TorchDtypeConcreteInputDesc, self).__init__()
        self.value = input_value
        assert isinstance(input_value, torch.dtype)
        self.dtype = torch.dtype

    def _create_value(self):
        return self.value

    def __str__(self):
        return "TDT{{{}}}".format(torch_dtype_to_str_dict[self.value])

    def to_yaml(self, yml_dir):
        input_config = {}
        input_config["get_func"] = GetFuncName_to_ConcreteInput_Reverse_Mappings[
            TorchDtypeConcreteInputDesc
        ]
        input_config["value_from"] = torch_dtype_to_str_dict[self.value]
        return input_config

    @staticmethod
    def from_yaml(input_config, yml_dir):
        return TorchDtypeConcreteInputDesc(
            str_to_torch_dtype_dict[input_config["value_from"]]
        )


class TorchSizeConcreteInputDesc(InputDescBase):
    """
    This class is used to depict an fixed input of type torch.dtype.
    torch.Size, a subclass of tuple
    """

    def __init__(self, input_value):
        """
        Args:
            input_values (list of torch tensor or numpy value or python list):
                The inputs that can be directly used by training without any post-processing.
                torch.Size, a subclass of tuple.
        """
        super(TorchSizeConcreteInputDesc, self).__init__()
        self.value = input_value
        assert isinstance(input_value, torch.Size)
        self.dtype = torch.Size

    def _create_value(self):
        return self.value

    def __str__(self):
        return "TSIZE{{{}}}".format(str(list(self.value)))

    def to_yaml(self, yml_dir):
        input_config = {}
        input_config["get_func"] = GetFuncName_to_ConcreteInput_Reverse_Mappings[
            TorchSizeConcreteInputDesc
        ]
        input_config["dtype"] = "torch.Size"
        input_config["value_from"] = str(list(self.value))
        return input_config

    @staticmethod
    def from_yaml(input_config, yml_dir):
        int_list = shape_from_str(input_config["value_from"])
        return TorchSizeConcreteInputDesc(torch.Size(int_list))


class TorchDeviceConcreteInputDesc(InputDescBase):
    """
    This class is used to depict an fixed input of type torch.dtype.

    """

    def __init__(self, input_value):
        """
        Args:
            input_values (list of torch tensor or numpy value or python list):
                The inputs that can be directly used by training without any post-processing.
        """
        super(TorchDeviceConcreteInputDesc, self).__init__()
        self.value = input_value
        assert isinstance(input_value, torch.device)
        self.dtype = torch.device

    def _create_value(self):
        return self.value

    def __str__(self):
        return "TD{{{}}}".format(str_from_torch_device(self.value))

    def to_yaml(self, yml_dir):
        input_config = {}
        input_config["get_func"] = GetFuncName_to_ConcreteInput_Reverse_Mappings[
            TorchDeviceConcreteInputDesc
        ]
        input_config["dtype"] = "torch.device"
        input_config["value_from"] = str_from_torch_device(self.value)
        return input_config

    @staticmethod
    def from_yaml(input_config, yml_dir):
        return TorchDeviceConcreteInputDesc(
            str_to_torch_device(input_config["value_from"])
        )


def torch_randn(shape, dtype, requires_grad, device):
    return torch.randn(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def torch_list_randn(shapes, dtypes, requires_grads, devices):
    return [
        torch_randn(s, dt, r, d)
        for s, dt, r, d in zip(shapes, dtypes, requires_grads, devices)
    ]


class TorchTensorLazyInputDesc(LazyInputDescBase):
    """
    This class is used to depict an input that can be allocated/randomized before the benchmark started.

    """

    def __init__(
        self, get_func_name: str, shape: list, dtype: torch.dtype, requires_grad, device
    ):
        super(TorchTensorLazyInputDesc, self).__init__(get_func_name, shape, dtype)
        assert isinstance(dtype, torch.dtype)
        assert (
            isinstance(get_func_name, str)
            and get_func_name in GetFuncName_to_LazyInputDesc_Mappings
        )
        self._requires_grad = requires_grad
        self._device = device

    def _create_value(self):
        with torch.no_grad():
            _, generator_func = GetFuncName_to_LazyInputDesc_Mappings[
                self._get_func_name
            ]
            return generator_func(
                self._shape, self._dtype, self._requires_grad, self._device
            )

    def __str__(self):
        return "TS{{{}}}".format(str(self._shape))

    def to_yaml(self, yml_dir):
        input_config = {}
        input_config["get_func"] = self._get_func_name
        input_config["value_from"] = str(self._shape)
        input_config["dtype"] = torch_dtype_to_str_dict[self._dtype]
        input_config["requires_grad"] = self._requires_grad
        input_config["device"] = str_from_torch_device(self._device)
        return input_config

    @staticmethod
    def from_yaml(input_config, yml_dir, get_func_name):
        return TorchTensorLazyInputDesc(
            get_func_name,
            shape_from_str(input_config["value_from"]),
            str_to_torch_dtype_dict[input_config["dtype"]],
            input_config["requires_grad"],
            str_to_torch_device(input_config["device"]),
        )


class TorchTensorListLazyInputDesc(LazyListInputDescBase):
    """
    This class is used to depict an tensor list input that can be allocated/randomized before the benchmark started.

    """

    def __init__(self, get_func_name, shapes, dtypes, requires_grads, devices):
        super(TorchTensorListLazyInputDesc, self).__init__(
            get_func_name, shapes, dtypes
        )
        assert isinstance(shapes, list)
        assert isinstance(dtypes, list)
        assert (
            isinstance(get_func_name, str)
            and get_func_name in GetFuncName_to_LazyInputDesc_Mappings
        )
        self._requires_grads = requires_grads
        self._devices = devices

    def _create_value(self):
        with torch.no_grad():
            _, generator_func = GetFuncName_to_LazyInputDesc_Mappings[
                self._get_func_name
            ]
            return generator_func(
                self._shapes, self._dtypes, self._requires_grads, self._devices
            )

    def __str__(self):
        return "TS_LIST{{{}}}".format(str(self._shapes))

    def to_yaml(self, yml_dir):
        input_config = {}
        input_config["get_func"] = self._get_func_name
        input_config["value_from"] = [str(s) for s in self._shapes]
        input_config["dtype"] = [torch_dtype_to_str_dict[d] for d in self._dtypes]
        input_config["requires_grad"] = self._requires_grads
        input_config["device"] = [str_from_torch_device(d) for d in self._devices]
        return input_config

    @staticmethod
    def from_yaml(input_config, yml_dir, get_func_name):
        return TorchTensorListLazyInputDesc(
            get_func_name,
            [shape_from_str(s) for s in input_config["value_from"]],
            [str_to_torch_dtype_dict[d] for d in input_config["dtype"]],
            input_config["requires_grad"],
            [str_to_torch_device(d) for d in input_config["device"]],
        )


#### Torch Input Desc Section ENDS ####


class GetFuncNames:
    # for lazy input generator
    TORCH_RANDN_GET = "torch_randn_get"
    TORCH_LIST_RANDN_GET = "torch_list_randn_get"
    NUMPY_RANDOM_GET = "np_random_get"
    NUMPY_RANDOM_BI_P0_5_GET = "np_rand_bi_p0_5_get"

    # for concrete input
    TORCH_TENSOR_VALUE_GET = "torch_tensor"
    TORCH_DTYPE_VALUE_GET = "torch_dtype"
    TORCH_DEVICE_VALUE_GET = "torch_device"
    TORCH_SIZE_VALUE_GET = "torch_size"
    NUMPY_VALUE_GET = "np_value"
    PYTHON_VALUE_GET = "py_value"


GetFuncName_to_LazyInputDesc_Mappings = {
    GetFuncNames.TORCH_RANDN_GET: [TorchTensorLazyInputDesc, torch_randn],
    GetFuncNames.TORCH_LIST_RANDN_GET: [TorchTensorListLazyInputDesc, torch_list_randn],
    GetFuncNames.NUMPY_RANDOM_GET: [NumpyLazyInputDesc, np_rand],
    GetFuncNames.NUMPY_RANDOM_BI_P0_5_GET: [NumpyLazyInputDesc, np_rand_bi_p0_5],
}

GetFuncName_to_ConcreteInput_Mappings = {
    GetFuncNames.NUMPY_VALUE_GET: NumpyConcreteInputDesc,
    GetFuncNames.TORCH_TENSOR_VALUE_GET: TorchTensorConcreteInputDesc,
    GetFuncNames.PYTHON_VALUE_GET: PythonTypeConcreteInputDesc,
    GetFuncNames.TORCH_DTYPE_VALUE_GET: TorchDtypeConcreteInputDesc,
    GetFuncNames.TORCH_DEVICE_VALUE_GET: TorchDeviceConcreteInputDesc,
    GetFuncNames.TORCH_SIZE_VALUE_GET: TorchSizeConcreteInputDesc,
}
GetFuncName_to_ConcreteInput_Reverse_Mappings = {
    y: x for x, y in GetFuncName_to_ConcreteInput_Mappings.items()
}


class InputDescSet:
    """
    This class is used to represent all inputs including both arg inputs and kwarg inputs.
    The orders are usually strictly aligned with model's forward function's inputs.

    """

    def __init__(self, *args_descs, **kwargs_descs):
        self.args_inputs = []
        for arg in args_descs:
            self.args_inputs.append(arg)

        self.kwargs_inputs = {}
        for k in kwargs_descs:
            self.kwargs_inputs[k] = kwargs_descs[k]


def calc_stat_items(data, prefix, stats, percentiles):
    # For cases, cannot run or fail to have backout results (due to failed run or, no required gradients)
    if len(data) == 0:
        data = [-1]
    stat_items = {
        f"{prefix}_{stat}": StatisticItem(
            getattr(np, stat)(data), is_basic_item=stat in ["mean"]
        )
        for stat in stats
    }

    if percentiles:
        percentiles_rets = np.quantile(data, percentiles).tolist()
        for index, r in enumerate(percentiles_rets):
            stat_items[f"{prefix}_p{percentiles[index]}"] = StatisticItem(
                r, is_basic_item=percentiles[index] in [0.8]
            )

    return stat_items


class StatisticItem:
    """
    This class is used to represent one single entry of statistic.

    """

    def __init__(self, value, is_diffable=True, is_basic_item=False):
        """
        Args:
            value (any type): statistic value (usually is a float).
            is_diffable (bool): whether this statistic value can be diff-able in comparision mode.
            is_basic_item (bool): whether this will be output in `show_basic_stat_item_only` mode defined in visual_config.
        """
        self._value = value
        self._is_diffable = is_diffable
        self._is_basic_item = is_basic_item


class RetRecordName:
    """
    This class is used to represent one single column properties.

    """

    def __init__(
        self,
        name,
        is_input=False,
        is_variable=False,
        is_statistic=False,
        is_diffable=False,
        is_basic_item_output=False,
    ):
        self._name = name
        self._is_input = is_input
        self._is_variable = is_variable
        self._is_statistic = is_statistic
        self._is_diffable = is_diffable
        self._is_basic_item_output = is_basic_item_output

    def __str__(self):
        return "name: {}, _is_input: {}, _is_variable: {}, _is_statistic: {}, _is_diffable: {}, _is_basic_item: {}".format(
            self._name,
            self._is_input,
            self._is_variable,
            self._is_statistic,
            self._is_diffable,
            self._is_basic_item_output,
        )


class RetRecordNameSet:
    """
    This class is used to be containers of multiple `RetRecordName`s

    """

    def __init__(self):
        self._name_set = []

    def extend(self, record_names):
        self._name_set.extend(record_names)

    def __iter__(self):
        for n in self._name_set:
            yield n


class RetRecordValue:
    """
    This class is used to represent one single column value.

    """

    def __init__(
        self,
        value,
    ):
        self._value = value

    def __str__(self):
        return "_value: {}".format(self._value)


class RetRecordValueSet:
    """
    This class is used to be containers of multiple `RetRecordValue`s

    """

    def __init__(self):
        self._value_set = []

    def extend(self, record_values):
        self._value_set.extend(record_values)

    def append(self, record_value):
        self._value_set.append(record_value)

    def __iter__(self):
        for _, v in enumerate(self._value_set):
            yield v


class RunRets:
    """
    This class is used to manage column names and all comulmn values for one benchmark run.

    """

    def __init__(self):
        # record keys
        self._record_names = None

        # record values
        self._record_values = []

        self._is_initialized = False

    def append(self, input_combination, variable_combination, statistic_items_in_dict):
        if not self._is_initialized:
            self._record_names = RetRecordNameSet()
            self._record_names.extend(
                list(
                    RetRecordName(input_name, is_input=True)
                    for input_name in input_combination.keys()
                )
            )

            self._record_names.extend(
                list(
                    RetRecordName(variable_name, is_variable=True)
                    for variable_name in variable_combination.keys()
                )
            )

            self._record_names.extend(
                list(
                    RetRecordName(
                        statistic_name,
                        is_statistic=True,
                        is_diffable=item._is_diffable,
                        is_basic_item_output=item._is_basic_item,
                    )
                    for statistic_name, item in statistic_items_in_dict.items()
                )
            )

        new_record_values = RetRecordValueSet()
        for record_name in self._record_names:
            if record_name._name in input_combination:
                assert record_name._is_input == True

                # for input values, we just save its shape and dtype.
                input_record_value = input_combination[record_name._name]
                if isinstance(input_record_value, np.ndarray):
                    input_record_value = npyarray2string(input_record_value)
                elif torch.is_tensor(input_record_value):
                    input_record_value = torch_tensor2string(input_record_value)

                new_record_values.append(RetRecordValue(input_record_value))
            elif record_name._name in variable_combination:
                assert record_name._is_variable == True
                new_record_values.append(
                    RetRecordValue(variable_combination[record_name._name])
                )
            elif record_name._name in statistic_items_in_dict.keys():
                assert record_name._is_statistic == True
                new_record_values.append(
                    RetRecordValue(statistic_items_in_dict[record_name._name]._value)
                )
            else:
                raise ValueError(
                    f'find invalid column: column name "{record_name._name}", input_combination {input_combination.keys()}, '
                    f"variable_combination {variable_combination.keys()}, statistic_items_in_dict {statistic_items_in_dict.keys()}"
                )

        self._record_values.append(new_record_values)

        if not self._is_initialized:
            self._is_initialized = True

    @property
    def record_names(self):
        return copy.deepcopy(self._record_names)

    def iterator(self):
        for values in self._record_values:
            yield {n: v for n, v in zip(self._record_names, values)}.items()

    def to_table_info(
        self,
        diff_view,
        show_basic_stat_item_only,
        pivot_variable_name=None,
        pivot_varible_control_value=None,
    ):
        """Generate statistics in format of data, header, sub_header.

        Args:
            diff_view (bool): whether generate the statisics in diff_view.
            show_basic_stat_item_only (bool): only return basic stat item.
            pivot_variable_name (str, optional): the variable name that is used to do the diff.
                Only applicable when diff_view is True. Defaults to None.
            pivot_varible_control_value (str, optional): the variable value that is used as baseline to do the diff.
                Only applicable when diff_view is True. Defaults to None.

        Returns:
            list of list: Each inner list is a result representing a single run.
            list: Each element is the category of the column. Only applicable for diff_view is True.
            list: Each element is the column of name.
        """
        if diff_view is True:
            return self._generate_table_info_for_diff_view(
                show_basic_stat_item_only,
                pivot_variable_name,
                pivot_varible_control_value,
            )
        else:
            return self._generate_table_info(show_basic_stat_item_only)

    def _generate_table_info_for_diff_view(
        self,
        show_basic_stat_item_only,
        pivot_variable_name,
        pivot_varible_control_value,
    ):
        # assert visual_config and visual_config.is_valid
        # show_basic_stat_item_only = visual_config.show_basic_stat_item_only
        sub_header = None
        group_by_pivot = {}
        # iterate each run result, group by two layers:
        #   layer 1: pivot viariable value
        #   layer 2: input values + variable values (excluding pivot) as key.
        #       This is needed to map between control and treatment compare.
        for record_name_value_pairs in self.iterator():
            input_values_and_excluding_pivot_variable_values = []
            pivot_value = None
            for record_name, record_value in record_name_value_pairs:
                if (
                    record_name._is_variable or record_name._is_input
                ) and record_name._name != pivot_variable_name:
                    value_str = record_value._value
                    if isinstance(record_value._value, np.ndarray):
                        value_str = npyarray2string(record_value._value)
                    input_values_and_excluding_pivot_variable_values.append(
                        str(value_str)
                    )

                if record_name._name == pivot_variable_name:
                    pivot_value = record_value._value

            if pivot_value is None:
                raise RuntimeError(
                    "pivot_value is none: {}".format(record_name_value_pairs)
                )

            if pivot_value not in group_by_pivot:
                group_by_pivot[pivot_value] = {}

            sub_key = "_".join(input_values_and_excluding_pivot_variable_values)

            group_by_pivot[pivot_value][sub_key] = record_name_value_pairs

        control = pivot_varible_control_value
        if control in group_by_pivot and len(group_by_pivot.keys()) != 2:
            raise RuntimeError("cannot use more than 2-values variable as pivot")

        povot_value_pool = list(group_by_pivot.keys())
        povot_value_pool.remove(control)
        treatment = povot_value_pool[0]

        # prepare the result headers, including top header (category for columns) + sub header (detailed columns)
        header_category = None
        header = []
        statistic_names = []
        diffable_statistic_names = []
        for record_name in self.record_names:
            if record_name._is_statistic:
                if not show_basic_stat_item_only or record_name._is_basic_item_output:
                    if record_name._is_diffable:
                        diffable_statistic_names.append(record_name._name)
                    statistic_names.append(record_name._name)

        header = ["input & run config"]

        # header_category = ['input & run config']
        header += [
            "{}.{}".format(control, n) for n in statistic_names
        ]  # for control group
        # header_category += len(statistic_names) * \
        #     ['control group - {} ({})'.format(control, 'us')]
        header += [
            "{}.{}".format(treatment, n) for n in statistic_names
        ]  # for treatment group
        # header_category += len(statistic_names) * \
        #     ['treatment group - {} ({})'.format(treatment, 'us')]
        header += [
            "{}.{}".format("diff", n) for n in diffable_statistic_names
        ]  # for diff group
        # header_category += len(diffable_statistic_names) * ['diff']

        # prepare the diff table body
        updated_data = []
        for sub_key, record_name_value_pairs in group_by_pivot[control].items():
            updated_row = []
            diff_control_colums = []
            var_and_input_row = []
            for record_name, record_value in record_name_value_pairs:
                if (
                    record_name._is_variable or record_name._is_input
                ) and record_name._name != pivot_variable_name:
                    value_str = record_value._value
                    if isinstance(record_value._value, np.ndarray):
                        value_str = npyarray2string(record_value._value)
                    elif torch.is_tensor(value_str):
                        value_str = torch_tensor2string(value_str)
                    var_and_input_row.append(str(value_str))

            updated_row = [";".join(var_and_input_row)]
            for record_name, record_value in record_name_value_pairs:
                if record_name._is_statistic:
                    if (
                        not show_basic_stat_item_only
                        or record_name._is_basic_item_output
                    ):
                        if record_name._is_diffable:
                            diff_control_colums.append(record_value._value)
                        updated_row.append(record_value._value)

            diff_treatment_colums = []
            for record_name, record_value in group_by_pivot[treatment][sub_key]:
                if record_name._is_statistic:
                    if (
                        not show_basic_stat_item_only
                        or record_name._is_basic_item_output
                    ):
                        if record_name._is_diffable:
                            diff_treatment_colums.append(record_value._value)
                        updated_row.append(record_value._value)

            from operator import truediv

            sub_ret = list(map(truediv, diff_treatment_colums, diff_control_colums))
            updated_row += ["{0:.3%}".format(s - 1.0) for s in sub_ret]
            updated_data.append(updated_row)

        return updated_data, header_category, header

    def _generate_table_info(self, show_basic_stat_item_only):
        header_category = None
        header = []

        # for record_name in self.record_names:
        # if (record_name._is_variable or record_name._is_input):
        #     var_and_input.append(record_name._name)
        var_and_input_names = []
        for record_name in self.record_names:
            if record_name._is_statistic:
                if not show_basic_stat_item_only or record_name._is_basic_item_output:
                    new_record_name = (
                        record_name._name + "({})".format("us")
                        if record_name._is_statistic and record_name._is_diffable
                        else record_name._name
                    )
                    header.append(new_record_name)
            elif record_name._is_variable or record_name._is_input:
                var_and_input_names.append(record_name._name)

        header = ["input & run config"] + header

        updated_data = []
        for record_name_value_pairs in self.iterator():
            updated_row = []
            var_and_input_row = []
            stat_row = []
            for record_name, record_value in record_name_value_pairs:
                if record_name._is_statistic:
                    if (
                        not show_basic_stat_item_only
                        or record_name._is_basic_item_output
                    ):
                        stat_row.append(record_value._value)
                elif record_name._is_variable or record_name._is_input:
                    if isinstance(record_value._value, np.ndarray):
                        value_str = npyarray2string(record_value._value)
                    elif torch.is_tensor(record_value._value):
                        value_str = torch_tensor2string(record_value._value)
                    else:
                        value_str = str(record_value._value)
                    var_and_input_row.append(str(value_str))
            updated_row = [";".join(var_and_input_row)] + stat_row
            updated_data.append(updated_row)
        return updated_data, header_category, header


def assert_values_are_close(input, other, rtol=1e-03, atol=1e-03):
    are_close = torch.allclose(input, other, rtol=rtol, atol=atol)
    if not are_close:
        abs_diff = torch.abs(input - other)
        abs_other = torch.abs(other)
        max_atol = torch.max((abs_diff - rtol * abs_other))
        max_rtol = torch.max((abs_diff - atol) / abs_other)
        err_msg = "The maximum atol is {}, maximum rtol is {}".format(
            max_atol, max_rtol
        )
        assert False, err_msg
