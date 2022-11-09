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
    PythonTypeConcreteInputDesc,
)

# -------------------------------
# torch.repeat (aka ONNX Expand + Tile) benchmark
# -------------------------------

repeat_configs = [
    BenchmarkConfig(
        input_desc_sets=[
            InputDescSet(
                TorchTensorLazyInputDesc(
                    GetFuncNames.TORCH_RANDN_GET,
                    [1, 64, 16, 32],
                    torch.float32,
                    True,
                    DEVICE,
                ),
                PythonTypeConcreteInputDesc([2, 1, 16, 1]),
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


@op_benchmark_with_report(repeat_configs, visual_config)
def bench_repeat(*args, **kwargs):
    class TileNet(torch.nn.Module):
        def forward(self, x, repeats):
            x = x.repeat(*repeats)
            return x

    return TileNet()


if __name__ == "__main__":
    bench_name = "repeat"
    bench_repeat.run(bench_name)
