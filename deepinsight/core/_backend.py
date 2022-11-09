import torch
from collections import abc


class Backends:
    OnnxRuntime = "ortmodule"
    PyTorch = "torch"
    ColumnName = "backend"


class Backend:
    def __init__(self, model, args_inputs, kwargs_inputs):
        self.type = None
        self.non_differentiable_forward_outputs = None
        self.model = model
        self.args = args_inputs
        self.kwargs = kwargs_inputs

    def execute_forward_path(self):
        return self._execute_forward_path()

    def set_non_differentiable_forward_output(self, list_of_index):
        self.non_differentiable_forward_outputs = list_of_index


# Flatten forward outputs using same way ORTModule did in orttraining/orttraining/python/training/ortmodule/_io.py
def _flatten_outputs(output, flattened_forward_outputs, output_idx):
    # Depth first traversal to traverse through the entire outputs
    if output is None:
        return
    elif isinstance(output, torch.Tensor):
        output_idx[0] += 1
        flattened_forward_outputs.append(output)
        return

    if isinstance(output, abc.Sequence):
        for value in output:
            _flatten_outputs(value, flattened_forward_outputs, output_idx)
    elif isinstance(output, abc.Mapping):
        for _, value in sorted(output.items()):
            _flatten_outputs(value, flattened_forward_outputs, output_idx)
    else:
        raise RuntimeError("unspported type")


class OnnxRuntimeBackend(Backend):
    def __init__(self, model, benchmark_name, args_inputs, kwargs_inputs):
        super(OnnxRuntimeBackend, self).__init__(model, args_inputs, kwargs_inputs)
        self.type = Backends.OnnxRuntime
        self._init(model, benchmark_name)

    def _init(self, model, benchmark_name):
        from onnxruntime.training.ortmodule import ORTModule, DebugOptions, LogLevel
        from onnxruntime.training.ortmodule._custom_autograd_function import (
            enable_custom_autograd_support,
        )

        enable_custom_autograd_support()
        self.model = ORTModule(
            model,
            debug_options=DebugOptions(
                # log_level=LogLevel.INFO,
                save_onnx=True,
                onnx_prefix=benchmark_name,
            ),
        )

    def _execute_forward_path(self):
        unflattened_forward_outputs = self.model(*self.args, **self.kwargs)

        # flattened outputs
        flattened_forward_outputs = []
        output_idx = [0]
        _flatten_outputs(
            unflattened_forward_outputs, flattened_forward_outputs, output_idx
        )
        return flattened_forward_outputs

    def get_non_differentiable_forward_outputs(self):
        if self.non_differentiable_forward_outputs is None:
            self.non_differentiable_forward_outputs = []
        return self.non_differentiable_forward_outputs

    def generate_non_differentiable_forward_outputs(self, *inputs, **kwargs):
        if self.non_differentiable_forward_outputs is None:
            non_differentiable_index_list = []
            # # enable grad because we want to make sure
            # # the forward run is in training mode
            # with torch.enable_grad():
            # Cannot call forward run, which will somehow make some torch.jit.script decorated function
            # compiled differently compared with storck PyTorch compilation.
            # forward_output_tuple = self.execute_forward_path()

            training_manager = (
                self.model._torch_module._execution_manager._training_manager
            )
            build_gradient_graph = training_manager._export_model(*inputs, **kwargs)

            if build_gradient_graph:
                # If model was exported, then initialize the graph builder
                training_manager._initialize_graph_builder(training=True)
            else:
                raise RuntimeError(
                    "fail to export model during generate_non_differentiable_forward_outputs"
                )

            # Build the gradient graph, self._graph_info is update in this phase.
            if build_gradient_graph:
                training_manager._build_graph()

            for (
                idx
            ) in training_manager._graph_info.output_grad_indices_non_differentiable:
                non_differentiable_index_list.append(idx)
            self.non_differentiable_forward_outputs = non_differentiable_index_list

        return self.non_differentiable_forward_outputs


class PyTorchBackend(Backend):
    def __init__(self, model, args_inputs, kwargs_inputs):
        super(PyTorchBackend, self).__init__(model, args_inputs, kwargs_inputs)
        self.type = Backends.PyTorch

    def get_non_differentiable_forward_outputs(self):
        if self.non_differentiable_forward_outputs is None:
            self.non_differentiable_forward_outputs = []
        return self.non_differentiable_forward_outputs

    def _execute_forward_path(self):
        forward_outputs = self.model(*self.args, **self.kwargs)

        flattened_forward_outputs = []
        output_idx = [0]
        _flatten_outputs(forward_outputs, flattened_forward_outputs, output_idx)

        # We make sure all outputs are contiguous, thus, comparable with ORT.
        # For backward, we don't need do this explicitly, because, all parameters and inputs are contigous,
        # so output grads will be aligned with them automatically.(https://pytorch.org/docs/stable/autograd.html#default-gradient-layouts)
        updated_flattened_forward_outputs = []
        for idx, y in enumerate(flattened_forward_outputs):
            if torch.is_tensor(y):
                updated_flattened_forward_outputs.append(
                    flattened_forward_outputs[idx].contiguous()
                )
            else:
                updated_flattened_forward_outputs.append(flattened_forward_outputs[idx])

        return updated_flattened_forward_outputs
