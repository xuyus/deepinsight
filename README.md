# DeepInsight

DeepInsight is a Python utility to provide a systematic way for benchmarking an individual operator/subgraph of a model with different scales of inputs.

## Features

Key features:

- Currently supported frameworks: PyTorch (`torch`), ONNX Runtime (`ortmodule`)
- Input: user-defined input values, randomly generated input values of user-defined shapes

An extrator is also provided which can dump input shape information for each operator of interest in a user model. Such information is helpful for further analysis including operator benchmarking.

## Usage

### Installation

Currently installing from source is supported.
```sh

pip install .
# or
python setup.py install
```

If user prefers installing an editable version, just change the intallation command into

```sh
pip install --editable .
# or
pip install -e .
# or
pip setup.py develop
```


### Define Benchmark Case In Python Code

Benchmarking cases are present in the `examples/benchmark_defs/` folder, supposed more cases are added there to enrich the operator/blocks coverage.

Simply run the *.py file will trigger the benchmark.

### Extract Benchmark Cases From Real Models

In terms of extraction scope, there are a few: Torch operator extraction, Torch NN.Module extraction, Torch NN.Module with hierachy extraction.

Check the examples folder for the patching code. A real model sample as below, extracted step 100 data here to exclude those inputs that are potentially NaN in the early training steps.

```python

  def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
    model.train()
    agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

    loss, sample_size, logging_output = criterion(model, sample["tlm"])

    print('fairseq_task update_num:{}'.format(update_num))
    # after forward run, add this check
    if update_num == 100:
        raise RuntimeError("Stop by intention")
  
    optimizer.backward(loss)

    # after backward, add this check
    if update_num == 99:
        from deepinsight.extractor import patching_block_and_operator_with_hierarchy_hook
        patching_block_and_operator_with_hierarchy_hook(model, False, "/tmp/sample_block_and_operator_hierarchy")

```

The extracted results will be generated to `export_dir`, which contains:
- a yaml file that include all benchmark descriptions, operator/NN.Module instance and inputs.
- [Optional] input data in format of (.pt, *npy) that is exported.
- [Optional] serialized module instance folder (only applicable for Torch NN.Module with hierachy extraction), the folder stores the dumpped module instances.

To Run the benchmark extracted, simply run command:

```sh
deepinsight input_dir=<path of directory storing one or more yaml files> output_dir=<target directory> name=<a string used to store outputs for this run>

other arguments:
  idx                    Each benchmark def in yaml file has a index, we can use this option to specify which benchmark to run.
  profile                Use Nsight system to do profiling for all runs. Profling result are put in folder `output_dir/name`.
  ort_only               Run with ONNX Runtime only, exclusive to `pt_only`.
  pt_only                Run with PyTorch only, exclusive to `ort_only`.
  excludes_idx_prefix    The benchmark defs in yaml file that are excluded.
```

It runs and compares performance of torch/ortmodule jobs, results will be saved in `.csv` file, check the std output log for the csv path.
