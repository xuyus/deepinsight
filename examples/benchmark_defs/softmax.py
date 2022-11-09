import torch
from deepinsight.core import (
    BenchmarkConfig,
    op_benchmark_with_report,
    VisualConfig,
    Backends,
    ComputeMode,
    InputDescSet,
    GetFuncNames,
    TorchTensorLazyInputDesc,
    DEVICE,
)

# -------------------------------
# softmax  benchmark
# -------------------------------

softmax_configs = [
    BenchmarkConfig(
        input_desc_sets=[
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [2400, 128],
                    torch.float32,
                    True,
                    DEVICE,
                ),
            ),
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [2400, 256],
                    torch.float32,
                    True,
                    DEVICE,
                ),
            ),
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [2400, 512],
                    torch.float32,
                    True,
                    DEVICE,
                ),
            ),
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [2400, 1024],
                    torch.float32,
                    True,
                    DEVICE,
                ),
            ),
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [2400, 2048],
                    torch.float32,
                    True,
                    DEVICE,
                ),
            ),
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [2400, 3072],
                    torch.float32,
                    True,
                    DEVICE,
                ),
            ),
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [2400, 4096],
                    torch.float32,
                    True,
                    DEVICE,
                ),
            ),
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [2400, 6144],
                    torch.float32,
                    True,
                    DEVICE,
                ),
            ),
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [2400, 12800],
                    torch.float32,
                    True,
                    DEVICE,
                ),
            ),
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [16 * 16 * 384, 384],
                    torch.float32,
                    True,
                    DEVICE,
                )
            ),
        ],
        variable_names=[Backends.ColumnName, ComputeMode.ColumnName],
        variable_values_pool=[
            [Backends.OnnxRuntime, Backends.PyTorch],
            [ComputeMode.FullPrecision, ComputeMode.MixedPrecision],
        ],
        extract_kernel_info=False,
        run_backward=True,
    )
]

visual_config = VisualConfig(
    pivot_variable_name=Backends.ColumnName,
    pivot_varible_control_value=Backends.PyTorch,
)


@op_benchmark_with_report(softmax_configs, visual_config)
def bench_softmax(*args, **kwargs):
    class SoftmaxNet(torch.nn.Module):
        def __init__(self):
            super(SoftmaxNet, self).__init__()
            self.m = torch.nn.Softmax(dim=1)

        def forward(self, x):
            y = self.m(x)
            return y

    return SoftmaxNet()


if __name__ == "__main__":
    bench_name = "softmax"
    bench_softmax.run(bench_name)
