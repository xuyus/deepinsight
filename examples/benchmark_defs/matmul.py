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
# matmul benchmark
# -------------------------------

matmul_configs = [
    BenchmarkConfig(
        input_desc_sets=[
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [8, 512, 4096],
                    torch.float32,
                    True,
                    DEVICE,
                ),
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [1024, 4096],
                    torch.float32,
                    True,
                    DEVICE,
                ),
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET, [1024], torch.float32, True, DEVICE
                ),
            )
        ],
        variable_names=[Backends.ColumnName, ComputeMode.ColumnName],
        variable_values_pool=[
            [Backends.OnnxRuntime, Backends.PyTorch],
            [ComputeMode.MixedPrecision],
        ],
        extract_kernel_info=False,
        run_backward=True,
    )
]

visual_config = VisualConfig(
    pivot_variable_name=Backends.ColumnName,
    pivot_varible_control_value=Backends.PyTorch,
)


@op_benchmark_with_report(matmul_configs, visual_config)
def bench_matmul(*args, **kwargs):
    class LinearNet(torch.nn.Module):
        def __init__(self):
            super(LinearNet, self).__init__()

        def forward(self, x, y, z):
            x = torch.nn.functional.linear(x, y, z)
            return x

    return LinearNet()


if __name__ == "__main__":
    bench_name = "matmul"
    bench_matmul.run(bench_name)
