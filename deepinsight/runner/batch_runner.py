import torch
import argparse
import os
import yaml
import numpy as np
from pandas import DataFrame
from collections import Counter, OrderedDict

from deepinsight.extractor import FuncCallConvention
from deepinsight.core import Backends, display_report, VisualConfig, DEVICE

from deepinsight.runner._entry_runner import EntryRunner

import torch.multiprocessing as mp

torch.manual_seed(1)

not_supporting_fp16 = ["cumsum"]
not_supporting_backward = ["nll_loss"]

# Temporarily ignore some functions
excluded_names = []
# excluded_names += ['addcmul', 'addcmul_', 'addcdiv', 'addcdiv_', 'bernoulli',
#                     'random', 'random_', 'sum'] # not supported in export
# excluded_names += ['min'] # shape inference fails
# excluded_names += ['add_']
# excluded_names += ['norm'] # occurs in grad clip
# excluded_names += ['mul_', 'embedding']
# excluded_names += ['expand_as', 'expand', 'squeeze'] # grad graph fails
# excluded_names += ['multi_head_attention_forward', 'layer_norm'] # related to input_generator
excluded_names += [
    "element_size",
    "pin_memory",
    "backward",
    "zeros_like",
    "empty",
]  # , 'split', 'zeros', 'ones', 'zeros_like', 'backward'] # useless?
# excluded_names += ['gather', 'masked_fill', 'masked_fill_']
excluded_names += ["override_torch_manual_seed"]


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Batch runner to run benchmarks of operators extracted from a file"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default=None,
        help="yaml file with extracted input info",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="output folder to save the batch run result",
    )
    parser.add_argument(
        "-e",
        "--extract_kernel",
        action="store_true",
        default=False,
        help="whether to extract kernel info as well",
    )
    parser.add_argument(
        "-b",
        "--backward",
        default=False,
        action="store_true",
        help="Whether to run backward also",
    )
    parser.add_argument(
        "--run_ort_only",
        default=False,
        action="store_true",
        help="Run with Onnx Runtime only.",
    )
    parser.add_argument(
        "--run_pt_only",
        default=False,
        action="store_true",
        help="Run with PyTorch only.",
    )
    parser.add_argument(
        "--idx", type=str, default=None, help="Use comma to specify multiple ids."
    )
    parser.add_argument(
        "--exclude_idx_prefix",
        type=str,
        default=None,
        help="Use comma to specify multiple ids.",
    )
    return parser.parse_args(args=args)


def get_input_info_yaml_files(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".yaml")]


def get_output_csv_path(dir, input_info_yaml_file):
    return os.path.join(dir, os.path.basename(input_info_yaml_file) + ".csv")


def run_entry(size, args, visual_config, input_config, index, save_csv):
    torch.cuda.set_device(DEVICE)

    summary_df = DataFrame()
    idx_col = []
    ent_col = []

    ent = FuncCallConvention.from_yaml(input_config, index, args.input_dir)
    bench_name = ent.id()
    print(
        ">>>" * 10,
        " running config: ",
        bench_name,
        "\n[",
        ent.index,
        "] ",
        ent.desc(brief=False),
        "\n",
        "<<<" * 10,
    )
    try:
        e = EntryRunner(ent, args.extract_kernel, args.backward)
        e.run(bench_name, run_ort_only=args.run_ort_only, run_pt_only=args.run_pt_only)
        if e.run_rets is not None:
            df = display_report(e.run_rets, visual_config)
            idx_col = f"{ent.index}"
            ent_col = ent.desc(brief=False)
            summary_df = summary_df.append(df, ignore_index=True)
            summary_df.insert(0, "index", idx_col)
            summary_df.insert(1, "fn(inputs)[signature]", ent_col)
    except Exception as ex:
        raise ex

    print("*" * 100)

    if save_csv:
        summary_df.to_csv(
            save_csv, mode="a", index=False, header=not os.path.exists(save_csv)
        )
        print(f"Summary appended to {save_csv}")


def main(args=None):
    args = parse_args(args)
    size = 1
    assert not (args.run_ort_only and args.run_pt_only)

    # where ORTModule export ONNX files.
    os.environ["ORTMODULE_SAVE_ONNX_PATH"] = args.output_dir
    # disable ORTModule fallback PyTorch runs
    os.environ["ORTMODULE_FALLBACK_POLICY"] = "FALLBACK_DISABLE"
    os.environ["ORTMODULE_FALLBACK_RETRY"] = "False"

    os.environ["ORTMODULE_ONNX_OPSET_VERSION"] = "12"

    if args.run_ort_only or args.run_pt_only:
        visual_config = None
    else:
        visual_config = VisualConfig(
            pivot_variable_name=Backends.ColumnName,
            pivot_varible_control_value=Backends.PyTorch,
            show_basic_stat_item_only=True,
        )
    not_collected_cases = []

    selected_ids = None
    ignored_id_prefixes = None
    if args.idx:
        selected_ids = args.idx.split(",")
        print("Running selected benchmark cases, with case id in ", selected_ids)
    elif args.exclude_idx_prefix:
        ignored_id_prefixes = args.exclude_idx_prefix.split(",")
        print(
            "Ignoring benchmark cases, with case id starts with ", ignored_id_prefixes
        )

    input_info_yaml_files = get_input_info_yaml_files(args.input_dir)
    for input_info_yaml_file in input_info_yaml_files:
        save_csv = get_output_csv_path(args.output_dir, input_info_yaml_file)
        if os.path.exists(save_csv):
            print(f"WARNNING: {save_csv} already exists, removing it now.")
            os.remove(save_csv)

        with open(input_info_yaml_file, "rb") as f:
            input_info_config = yaml.full_load(f)
            for index, input_config in input_info_config.items():
                if selected_ids and index not in selected_ids:
                    continue
                if ignored_id_prefixes and any(
                    [index.startswith(p) for p in ignored_id_prefixes]
                ):
                    continue

                if (
                    input_config["type"] == "operator"
                    and input_config["func_name"] in excluded_names
                ):
                    continue

                try:
                    # do the benchmark in a new process in case different cases affect each other.
                    mp.spawn(
                        run_entry,
                        nprocs=size,
                        args=(args, visual_config, input_config, index, save_csv),
                    )
                except Exception as ex:
                    # raise ex
                    print(ex)
                    not_collected_cases.append((index, ex))

    if not_collected_cases != []:
        print(
            "Not collected cases:",
            *[f"{index}: {ex}" for index, ex in not_collected_cases],
            sep="\n",
        )


if __name__ == "__main__":
    main()
