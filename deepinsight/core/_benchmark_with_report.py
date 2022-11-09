import os, sys
from deepinsight.core._benchmark import BenchmarkRunner
from deepinsight.core._data import generate_file_name
import pandas as pd


def op_benchmark_with_report(benchmark_config, visual_config):
    """
    A function decorator for benchmarking and generate reports. The benchmark can then be executed by `.run`
    method on the return value.

    Args:
        benchmark_config (BenchmarkConfig): Benchmarking configuration.
        visual_config (VisualConfig): Report visualization configuration.
    """

    def wrapper(fn):
        return BenchmarkWithReportRunner(fn, benchmark_config, visual_config)

    return wrapper


class BenchmarkWithReportRunner(BenchmarkRunner):
    """
    The class manages the benchmark running, result saving.

    """

    def __init__(self, fn, benchmark_config, visual_config):
        super().__init__(fn, benchmark_config)
        self.visual_config = visual_config

    def run(self, name, save_csv=None, run_ort_only=False, run_pt_only=False):
        super().run(name, run_ort_only, run_pt_only)
        for run_rets in self.run_rets_collection:
            df = display_report(run_rets, self.visual_config)
            if not save_csv:
                df.to_csv(sys.stdout)
            else:
                df.to_csv(
                    save_csv, mode="a", index=False, header=not os.path.exists(save_csv)
                )
                print("Summary appended to file {}".format(save_csv))


class VisualConfig:
    """
    This class is used to config for stats visualization.

    """

    def __init__(
        self,
        pivot_variable_name=None,
        pivot_varible_control_value=None,
        show_basic_stat_item_only=True,
    ):
        """
        Args:
            pivot_variable_name (string):
                The variable we planned to use as comparasion pivot.
            pivot_varible_control_value (string):
                The value we take as baseline to do the compare.
        """
        self.pivot_variable_name = pivot_variable_name
        self.pivot_varible_control_value = pivot_varible_control_value
        self.show_basic_stat_item_only = show_basic_stat_item_only

    @property
    def is_valid(self):
        return self.pivot_variable_name and self.pivot_varible_control_value


def _to_pandas_frame(updated_data, header_category, header):
    if header_category is not None:
        index = [header_category, header]
    else:
        index = header
    df = pd.DataFrame(data=updated_data, columns=index)
    # print('=' * 100, "\nstats from {}: \n".format(self.benchmark_name), df.to_string(index=False, justify='left'))
    return df


def display_report(run_rets, visual_config=None):
    diff_view = False
    if visual_config is not None and visual_config.is_valid:
        diff_view = True
        updated_data, header_category, header = run_rets.to_table_info(
            diff_view,
            visual_config.show_basic_stat_item_only,
            pivot_variable_name=visual_config.pivot_variable_name,
            pivot_varible_control_value=visual_config.pivot_varible_control_value,
        )
    else:
        show_basic_stat_item_only = False
        updated_data, header_category, header = run_rets.to_table_info(
            diff_view, show_basic_stat_item_only
        )
    return _to_pandas_frame(updated_data, header_category, header)
