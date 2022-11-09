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

nonzero_configs = [
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


@op_benchmark_with_report(nonzero_configs, visual_config)
def bench_nonzero(*args, **kwargs):
    class NonZeroNet(torch.nn.Module):
        def __init__(self):
            super(NonZeroNet, self).__init__()

        def forward(self, x):
            x = torch.nonzero(x)
            return x

    return NonZeroNet()


if __name__ == "__main__":
    bench_name = "nonzero"
    bench_nonzero.run(bench_name)
