import time
import torch
import nvtx

from enum import Enum


def sync():
    torch.cuda.synchronize()


class TimerBase:
    """
    The base class for different Timer implementations.

    """

    def __init__(self, backend):
        self._backend = backend

    def run_forward(self):
        return self._run_forward()

    def run_backward(self, loss):
        self._run_backward(loss)

    def finalize(self):
        self._finalize()

    def get_durations(self):
        return self._get_durations()


class PerIterationSyncTimer(TimerBase):
    def __init__(self, backend, repeat_step, loss_func):
        super(PerIterationSyncTimer, self).__init__(backend)
        self._fw_start_time = [0 for _ in range(repeat_step)]
        self._fw_end_time = [0 for _ in range(repeat_step)]
        self._fw_step = 0

        self._bw_start_time = [0 for _ in range(repeat_step)]
        self._bw_end_time = [0 for _ in range(repeat_step)]
        self._bw_step = 0

    def _run_forward(self):
        sync()
        self._fw_start_time[self._fw_step] = time.time()
        y = self._backend.execute_forward_path()
        sync()
        self._fw_end_time[self._fw_step] = time.time()
        self._fw_step += 1
        return y

    def _run_backward(self, loss):
        sync()
        self._bw_start_time[self._bw_step] = time.time()
        loss.backward()
        sync()
        self._bw_end_time[self._bw_step] = time.time()
        self._bw_step += 1

    def _get_durations(self):
        rets = {"fw": [], "bw": []}
        for s, e in zip(self._fw_start_time, self._fw_end_time):
            rets["fw"].append((e - s) * 1e6)  # t s -> 1,000,000*t us
        for s, e in zip(self._bw_start_time, self._bw_end_time):
            rets["bw"].append((e - s) * 1e6)  # t s -> 1,000,000*t us
        return rets


class RecordBasedTimerBase(TimerBase):
    def __init__(self, backend, repeat_step, loss_func):
        super(RecordBasedTimerBase, self).__init__(backend)
        self._fw_start_event = [
            torch.cuda.Event(enable_timing=True) for _ in range(repeat_step)
        ]
        self._fw_end_event = [
            torch.cuda.Event(enable_timing=True) for _ in range(repeat_step)
        ]
        self._fw_step = 0

        self._bw_start_event = [
            torch.cuda.Event(enable_timing=True) for _ in range(repeat_step)
        ]
        self._bw_end_event = [
            torch.cuda.Event(enable_timing=True) for _ in range(repeat_step)
        ]
        self._bw_step = 0

        self._loss_func = loss_func

    def start_forward_trace(self):
        # record time of `fn`
        self._fw_start_event[self._fw_step].record()
        return self._fw_step

    def end_forward_trace(self, step):
        assert self._fw_step == step
        self._fw_end_event[step].record()
        self._fw_step += 1

    def start_backward_trace(self):
        self._bw_start_event[self._bw_step].record()
        return self._bw_step

    def end_backward_trace(self, step):
        assert self._bw_step == step
        self._bw_end_event[step].record()
        self._bw_step += 1

    def _finalize(self):
        # record clocks
        sync()

    def _get_durations(self):
        rets = {"fw": [], "bw": []}
        if self._fw_step > 0:
            for s, e in zip(self._fw_start_event, self._fw_end_event):
                rets["fw"].append(s.elapsed_time(e) * 1e3)  # ms -> us
        if self._bw_step > 0:
            for s, e in zip(self._bw_start_event, self._bw_end_event):
                rets["bw"].append(s.elapsed_time(e) * 1e3)
        return rets


class ORTRecordBasedTimer(RecordBasedTimerBase):
    def __init__(self, backend, repeat_step, loss_func):
        super(ORTRecordBasedTimer, self).__init__(backend, repeat_step, loss_func)

        # Patch original `run_forward()` of `TrainingAgent`
        from onnxruntime.training.ortmodule._training_manager import TrainingAgent

        def patched_run_forward(target, feeds, fetches, state, cache=None):
            current_step = self.start_forward_trace()
            with nvtx.annotate(message="ORT_FOWARD", color="yellow"):
                target._training_agent.run_forward(feeds, fetches, state, cache)
            self.end_forward_trace(current_step)

        TrainingAgent.run_forward = patched_run_forward

        def patched_run_backward(target, feeds, fetches, state):
            current_step = self.start_backward_trace()
            with nvtx.annotate(message="ORT_BACKWARD", color="red"):
                target._training_agent.run_backward(feeds, fetches, state)
            self.end_backward_trace(current_step)

        TrainingAgent.run_backward = patched_run_backward

        def custom_hook(module, grad_input, grad_output):
            # we make it contiguous if not, to make sure
            # ORT run is comparable with Torch run.
            grad_input = [
                grad.contiguous() if grad is not None else None for grad in grad_input
            ]
            return tuple(grad_input)

        self._loss_func.register_full_backward_hook(custom_hook)

    def _run_forward(self):
        return self._backend.execute_forward_path()

    def _run_backward(self, loss):
        loss.backward()

    def _finalize(self):
        super()._finalize()
        from onnxruntime.training.ortmodule._training_manager import TrainingAgent

        def original_run_forward(target, feeds, fetches, state, cache=None):
            target._training_agent.run_forward(feeds, fetches, state, cache)

        TrainingAgent.run_forward = original_run_forward

        def original_run_backward(target, feeds, fetches, state):
            target._training_agent.run_backward(feeds, fetches, state)

        TrainingAgent.run_backward = original_run_backward


class PTRecordBasedTimer(RecordBasedTimerBase):
    def __init__(self, backend, repeat_step, loss_func):
        super(PTRecordBasedTimer, self).__init__(backend, repeat_step, loss_func)
        self.rng = None

        def custom_hook(module, grad_input, grad_output):
            grad_input = [
                grad.contiguous() if grad is not None else None for grad in grad_input
            ]
            self.start_backward_trace()
            self.rng = nvtx.start_range(message="PT_BACKWARD", color="blue")
            return tuple(grad_input)

        self._loss_func.register_full_backward_hook(custom_hook)

    def _run_forward(self):
        step = self.start_forward_trace()
        with nvtx.annotate(message="PT_FOWARD", color="green"):
            y = self._backend.execute_forward_path()
        self.end_forward_trace(step)
        return y

    def _run_backward(self, loss):
        loss.backward()
        nvtx.end_range(self.rng)
        self.end_backward_trace(self._bw_step)

    def _finalize(self):
        super()._finalize()


class TimerType(Enum):
    EventRecordORT = 1
    EventRecordPT = 2
    PerIterationSync = 3
    RealE2ESync = 4


class TimerFactory:
    def __init__(self):
        super(TimerFactory, self).__init__()
        self._Timers = {
            TimerType.EventRecordORT: ORTRecordBasedTimer,
            TimerType.EventRecordPT: PTRecordBasedTimer,
            TimerType.PerIterationSync: PerIterationSyncTimer,
            # TimerType.RealE2ESync: None
        }

    def create_timer(self, time_type, backend, repeat_step, loss_func):
        it = self._Timers.get(time_type, None)
        if not it:
            raise RuntimeError("unrecognized timer type {}".format(time_type))
        return self._Timers[time_type](backend, repeat_step, loss_func)
