import torch
from deepinsight.extractor import (
    FuncCallConvention,
    OperatorCallConvention,
    BlockCallConvention,
    BlockHierarchyCallConvention,
)
from deepinsight.core import (
    InputDescSet,
    BenchmarkConfig,
    BenchmarkRunner,
    ComputeMode,
    Backends,
    DEVICE,
)


class EntryRunner:
    def __init__(
        self, entry: FuncCallConvention, extract_kernel_info=False, run_backward=False
    ):
        self.is_block = True
        if isinstance(entry, OperatorCallConvention):
            self.is_block = False
        elif isinstance(entry, BlockCallConvention):
            self.is_block = True
        elif isinstance(entry, BlockHierarchyCallConvention):
            self.is_block = True
        else:
            raise RuntimeError("invalid function call convention type ", entry)

        self._func_call_convention = entry
        self.run_rets = None

        self.extract_kernel_info = extract_kernel_info
        self.run_backward = run_backward

    def run(self, bench_name, run_ort_only=False, run_pt_only=False):
        run_backward = self.run_backward
        input_configs = [
            BenchmarkConfig(
                input_desc_sets=[
                    InputDescSet(
                        *self._func_call_convention.func_inp_args,
                        **self._func_call_convention.func_inp_kwargs
                    )
                ],
                variable_names=[Backends.ColumnName, ComputeMode.ColumnName, "timing"],
                variable_values_pool=[
                    [Backends.PyTorch, Backends.OnnxRuntime],
                    [ComputeMode.MixedPrecision],
                    ["event_records"],
                ],
                extract_kernel_info=self.extract_kernel_info,
                run_backward=run_backward,
            )
        ]

        def bench_sample(*args, **kwargs):
            if self.is_block is True:

                class Net(torch.nn.Module):
                    def __init__(net_self):
                        super().__init__()
                        # need put the instance here for torch to
                        # understand the relationship between nn.Modules.
                        # todo: don't hard code cuda???
                        net_self.instance = self._func_call_convention.module
                        net_self.instance.to(DEVICE)

                    def forward(net_self, *args, **input_kwargs):
                        return net_self.instance(*args, **input_kwargs)

                return Net()
            else:
                # We don't pass non-tensor inputs as Net Module inputs, because non-tensor inpus are potentially be used by
                # exporter as constant (for example, transpose, permute, squeeze, unsqueeze).
                # Otherwise, we will have following error:
                #     File "/opt/conda/lib/python3.7/site-packages/torch/onnx/symbolic_helper.py", line 94, in _parse_arg
                #         "for argument '{}' of node '{}', got '{}'.".format(arg_name, node_name, value.node().kind()))
                #     RuntimeError: Expected node type 'onnx::Constant' for argument 'dim' of node 'unsqueeze', got 'prim::Param'.

                non_tensor_input_args_index = []
                for idx, arg in enumerate(args):
                    if torch.is_tensor(arg):
                        pass
                    elif isinstance(arg, torch.dtype):
                        pass
                    elif isinstance(arg, torch.Size):
                        pass
                    elif isinstance(arg, torch.device):
                        pass
                    else:
                        non_tensor_input_args_index.append(idx)

                non_tensor_input_kwargs_name = []
                for k, arg in kwargs.items():
                    if torch.is_tensor(arg):
                        pass
                    elif isinstance(arg, torch.dtype):
                        pass
                    elif isinstance(arg, torch.Size):
                        pass
                    elif isinstance(arg, torch.device):
                        pass
                    else:
                        non_tensor_input_kwargs_name.append(k)

                class Net(torch.nn.Module):
                    def __init__(net_self):
                        super().__init__()
                        net_self.func = self._func_call_convention.func

                    def forward(net_self, *net_args, **net_kwargs):
                        import copy

                        # ORTModule handle all inputs as torch tensor, here we try mapping the type back to original type.
                        # For safety, we do a deepcopy for those non torch data types.
                        new_args = [
                            copy.deepcopy(args[idx])
                            if idx in non_tensor_input_args_index
                            else arg
                            for idx, arg in enumerate(net_args)
                        ]
                        new_kwargs = {
                            k: arg
                            if k not in non_tensor_input_kwargs_name
                            else copy.deepcopy(kwargs[k])
                            for k, arg in net_kwargs.items()
                        }
                        return net_self.func(*new_args, **new_kwargs)

                return Net()

        bench_runner = BenchmarkRunner(bench_sample, input_configs)
        bench_runner.run(bench_name, run_ort_only=run_ort_only, run_pt_only=run_pt_only)

        # we only define one single benchmark, so get the first one
        self.run_rets = bench_runner.run_rets_collection[0]
