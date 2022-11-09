import os
import torch
import numpy
import itertools
from deepinsight.core._data import calc_stat_items, StatisticItem, RunRets
from deepinsight.core._timer import TimerFactory, TimerType
from deepinsight.core._backend import Backends, PyTorchBackend, OnnxRuntimeBackend
from deepinsight.core._loss_func import SimpleLossModule

DEVICE = torch.device("cuda:0")


class ComputeMode:
    FullPrecision = "fp32"
    MixedPrecision = "fp16"
    ColumnName = "mode"


def op_benchmark(benchmark_config):
    """
    A function decorator for benchmarking. The benchmark can then be executed by `.run`
    method on the return value.

    Args:
        benchmarks (BenchmarkConfig): Benchmarking configuration.
    """

    def wrapper(fn):
        return BenchmarkRunner(fn, benchmark_config)

    return wrapper


class BenchmarkConfig:
    """
    This class is used by the `op_benchmark` function.

    """

    def __init__(
        self,
        input_desc_sets,
        variable_names,
        variable_values_pool,
        extract_kernel_info=False,
        run_backward=False,
    ):
        """
        Args:
            input_desc_sets (a list of InputDescSet):
                A list containing all possible input values's description (for both arg inputs and kwarg inputs).
            variable_names (list string):
                The argument name of variables in this run.
            variable_values_pool (list of any type):
                All possible values of the variables in this run.
            extract_kernel_info (bool): Whether to dump kernel information (name/grid/block/dur) for each case.
            run_backward (bool): Whether to run backward pass for the model.

        """
        self.input_desc_sets = input_desc_sets

        self.variable_names = variable_names
        self.variable_values_pool = variable_values_pool
        self.extract_kernel_info = extract_kernel_info
        self.run_backward = run_backward


class BenchmarkRunner:
    """
    The class manages the benchmark running, result saving.

    """

    def __init__(self, fn, benchmark_configs):
        self.fn = fn
        self.benchmark_configs = benchmark_configs
        self.run_rets_collection = []

    def _generate_variable_combinations(self, bench):
        names = bench.variable_names
        vals = bench.variable_values_pool
        settings = []
        for name, val in zip(names, vals):
            setting = []
            for v in val:
                setting.append((name, v))
            settings.append(setting)
        cartesian_prod = itertools.product(*settings)
        return [dict(prod) for prod in cartesian_prod]

    def _prepare_backend_and_inputs(
        self, input_desc_set, one_variables_combination, benchmark_name
    ):
        mixed_precision_mode = (
            one_variables_combination[ComputeMode.ColumnName]
            == ComputeMode.MixedPrecision
        )
        backend = one_variables_combination[Backends.ColumnName]

        updated_args_inputs = [arg.get_data() for arg in input_desc_set.args_inputs]
        updated_kwargs_inputs = {
            k: v.get_data() for k, v in input_desc_set.kwargs_inputs.items()
        }
        if mixed_precision_mode:
            updated_args_inputs = [
                arg.half()
                if torch.is_tensor(arg) and arg.dtype in [torch.float32, numpy.float32]
                else arg
                for arg in updated_args_inputs
            ]

            updated_kwargs_inputs = {
                k: (
                    v.half()
                    if torch.is_tensor(v) and v.dtype in [torch.float32, numpy.float32]
                    else v
                )
                for k, v in updated_kwargs_inputs.items()
            }

        # This is just used to track input in final reports.
        input_args = {}
        for index, arg in enumerate(updated_args_inputs):
            input_args["input_{}".format(index)] = arg
        for input_name, input_value in updated_kwargs_inputs.items():
            input_args[input_name] = input_value

        model = self.fn(*updated_args_inputs, **updated_kwargs_inputs)
        model.train()
        with torch.no_grad():
            model = model.half() if mixed_precision_mode else model
            model = model.to(DEVICE)

        _backend = None
        if backend == Backends.OnnxRuntime:
            _backend = OnnxRuntimeBackend(
                model, benchmark_name, updated_args_inputs, updated_kwargs_inputs
            )
        elif backend == Backends.PyTorch:
            _backend = PyTorchBackend(model, updated_args_inputs, updated_kwargs_inputs)
        else:
            raise RuntimeError(f"unsupported backend {backend}")

        return _backend, input_args, updated_args_inputs, updated_kwargs_inputs

    def _run_iteration(
        self, backend, one_variables_combination, bench, warmup_step=20, repeat_step=100
    ):
        timing_approach = (
            one_variables_combination["timing"]
            if "timing" in one_variables_combination
            else "event_records"
        )
        ret_in_dict = run_op_benchmark(
            backend=backend,
            timing=timing_approach,
            run_backward=bench.run_backward,
            extract_kernel_info=bench.extract_kernel_info,
            repeat_step=repeat_step,
            warmup_step=warmup_step,
        )

        return ret_in_dict

    def _run(self, benchmark_name, bench, run_ort_only=False, run_pt_only=False):
        combination_list = self._generate_variable_combinations(bench)
        # print("all combinations listed as below:", combination_list)

        grad_output_index = []
        if run_pt_only is True:
            # skip when we only run PyTorch backend.
            pass
        else:
            # Run PyTorch first to make sure we have same path for torch.jit.script functions
            for input_desc_set in bench.input_desc_sets:
                for one_variables_combination in combination_list:
                    if (
                        one_variables_combination[Backends.ColumnName]
                        == Backends.PyTorch
                    ):
                        (
                            backend,
                            input_args,
                            updated_args_inputs,
                            updated_kwargs_inputs,
                        ) = self._prepare_backend_and_inputs(
                            input_desc_set, one_variables_combination, benchmark_name
                        )
                        self._run_iteration(
                            backend,
                            one_variables_combination,
                            bench,
                            warmup_step=0,
                            repeat_step=2,
                        )
                        break
                    else:
                        continue

            if bench.run_backward:
                for input_desc_set in bench.input_desc_sets:
                    for one_variables_combination in combination_list:
                        if (
                            one_variables_combination[Backends.ColumnName]
                            == Backends.OnnxRuntime
                        ):
                            (
                                backend,
                                _,
                                updated_args_inputs,
                                updated_kwargs_inputs,
                            ) = self._prepare_backend_and_inputs(
                                input_desc_set,
                                one_variables_combination,
                                benchmark_name,
                            )
                            grad_output_index = (
                                backend.generate_non_differentiable_forward_outputs(
                                    *updated_args_inputs, **updated_kwargs_inputs
                                )
                            )
                            # print(f"non_differentiable_forward_outputs_index_list: {grad_output_index}")
                            break
                        else:
                            continue

        run_rets = RunRets()
        allowed_backend = [Backends.OnnxRuntime, Backends.PyTorch]
        if run_ort_only is True:
            allowed_backend = [Backends.OnnxRuntime]
        if run_pt_only is True:
            allowed_backend = [Backends.PyTorch]

        for input_desc_set in bench.input_desc_sets:
            for one_variables_combination in combination_list:
                if one_variables_combination[Backends.ColumnName] in allowed_backend:
                    backend, input_args, _, _ = self._prepare_backend_and_inputs(
                        input_desc_set, one_variables_combination, benchmark_name
                    )
                    if bench.run_backward:
                        backend.set_non_differentiable_forward_output(grad_output_index)
                    ret_in_dict = self._run_iteration(
                        backend, one_variables_combination, bench
                    )
                    run_rets.append(input_args, one_variables_combination, ret_in_dict)

        return run_rets

    def run(self, name, run_ort_only=False, run_pt_only=False):
        has_single_bench = isinstance(self.benchmark_configs, BenchmarkConfig)
        benchmarks = (
            [self.benchmark_configs] if has_single_bench else self.benchmark_configs
        )

        for bench in benchmarks:
            run_rets = self._run(
                name, bench, run_ort_only=run_ort_only, run_pt_only=run_pt_only
            )
            self.run_rets_collection.append(run_rets)


def sync():
    torch.cuda.synchronize()


def run_op_benchmark(
    backend,
    run_backward=True,
    timing="event_records",
    extract_kernel_info=False,
    warmup_step=20,
    repeat_step=100,
):
    """Run operator computation representation fn `repeat_step` times, and generate kernel latency statistics.

    To minimize the cold start impact, we allow ignoring the initial `warmup_step` steps in out statistics.

    Args:
        run_backward (bool): Whether to run backward pass for the model.
        timing_fn (str): Timing function to measure time elapsed in training steps.
        extract_kernel_info (bool): Whether to dump kernel information (name/grid/block/dur) for each case.
        warmup_step (int): How many initial steps are NOT included in final statistics.
        repeat_step (int): How many steps are used for statistics.
        grad_to_none: (optional, list of gradients) List of gradients that are not intented to be accumulated
            in case of the overhead affecting kernel time measurement.

    Returns:
        Returns a dictionary (stat name, stat value):
            The first item in the dict is always the mean run time.
            The other items return corresponding performance percentiles,
                aligned with the input 'percentiles'.
    """
    print(
        f"run_op_benchmark: backend - {backend.type}, type(backend.model) - {type(backend.model)}"
    )

    # A list indicating the performance percentile matrix.
    # For example [0.2, 0.8] means, return 20-th and 80-th performance percentile.
    percentiles = [0.5, 0.8]
    output_min, output_max = True, True
    output_loss_validness = False

    # PyTorch Profiler profiling steps
    if extract_kernel_info:
        # print("profiling using PyTorch Profiler...")
        from torch.profiler import profile, ProfilerActivity, schedule
        import json
        from tempfile import NamedTemporaryFile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=1),
        ) as prof:
            for _ in range(3):
                sync()
                backend.execute_forward_path()
                sync()
                prof.step()
                sync()
        with NamedTemporaryFile("w+t") as f:
            # Export tracing info to a temp JSON file and parse kernel info.
            # Temp file is auto-deleted afterwards.
            prof.export_chrome_trace(f.name)
            tracing_events = json.load(open(f.name))["traceEvents"]
        kernel_events = [
            evt for evt in tracing_events if "cat" in evt and evt["cat"] == "Kernel"
        ]

    loss_func = SimpleLossModule()

    # warm up
    for _ in range(warmup_step):
        forward_outputs = backend.execute_forward_path()
        if run_backward:

            with torch.no_grad():
                bw_variables = []
                for idx, fw_output in enumerate(forward_outputs):
                    if (
                        torch.is_tensor(fw_output)
                        and fw_output.requires_grad
                        and idx not in backend.get_non_differentiable_forward_outputs()
                    ):
                        bw_variables.append(fw_output)

            loss = loss_func(*bw_variables)

            if len(bw_variables) > 0:
                loss.backward()

    timer_factory = TimerFactory()
    if timing == "event_records":
        time_type = (
            TimerType.EventRecordORT
            if backend.type == Backends.OnnxRuntime
            else TimerType.EventRecordPT
        )
    else:
        time_type = TimerType.PerIterationSync

    timer_instance = timer_factory.create_timer(
        time_type, backend, repeat_step, loss_func
    )
    stream_holder = torch.empty(int(256 << 20), dtype=torch.int8, device="cuda")
    loss_values = []
    for i in range(repeat_step):
        # we don't want `fn` to accumulate gradient values if it contains a backward pass. So we clear the
        # provided gradients
        for p in backend.model.parameters():
            if getattr(p, "grad") is not None:
                p.grad = None

        for i, arg in enumerate(backend.args):
            if torch.is_tensor(arg) and getattr(arg, "grad") is not None:
                backend.args[i].grad = None

        for k, arg in backend.kwargs.items():
            if torch.is_tensor(arg) and getattr(arg, "grad") is not None:
                backend.kwargs[k].grad = None

        stream_holder.zero_()
        forward_outputs = timer_instance.run_forward()
        with torch.no_grad():

            bw_variables = []
            for idx, fw_output in enumerate(forward_outputs):
                if (
                    torch.is_tensor(fw_output)
                    and fw_output.requires_grad
                    and idx not in backend.get_non_differentiable_forward_outputs()
                ):
                    bw_variables.append(fw_output)

        loss = loss_func(*bw_variables)

        if loss is None:
            loss_values.append(-1.0)
        else:
            loss_values.append(0.0)

        # print("backend: ", backend.type, "loss: ", loss)
        if run_backward and len(bw_variables) > 0:
            stream_holder.zero_()
            timer_instance.run_backward(loss)

    timer_instance.finalize()
    times = timer_instance.get_durations()
    fw_times = sorted(times["fw"])

    stats = ["mean", "std"]
    if output_min:
        stats.append("min")
    if output_max:
        stats.append("max")

    ret = calc_stat_items(fw_times, "fw", stats, percentiles)
    if extract_kernel_info:
        kernel_sv = StatisticItem(
            "-->".join(
                [
                    f'{evt["name"]}, grid {evt["args"]["grid"]}, block {evt["args"]["block"]}, dur {evt["dur"]}us'
                    for evt in kernel_events
                ]
            ),
            is_diffable=False,
            is_basic_item=True,
        )
        ret["kernel"] = kernel_sv

    if run_backward:
        bw_times = times["bw"]
        bw_times = sorted(bw_times)
        ret.update(calc_stat_items(bw_times, "bw", stats, percentiles))

    if output_loss_validness:
        loss_values_tensor = torch.tensor(loss_values)
        loss_values_tensor = torch.where(
            torch.isnan(loss_values_tensor),
            torch.full_like(loss_values_tensor, 0.0),
            torch.full_like(loss_values_tensor, 1.0),
        )
        loss_values_tensor = torch.where(
            torch.isinf(loss_values_tensor),
            torch.full_like(loss_values_tensor, 0.0),
            torch.full_like(loss_values_tensor, 1.0),
        )
        valid_loss_count = loss_values_tensor.sum()
        ret["loss_validness"] = StatisticItem(
            float(valid_loss_count) / (len(loss_values) + 1e-5), is_diffable=False
        )
    return ret
