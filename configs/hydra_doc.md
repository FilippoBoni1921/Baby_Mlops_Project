# Hydra Sweeps and Multirun Guide

## Overview

Hydra provides powerful tools to run your applications multiple times with different configurations. This is essential for tasks like hyperparameter tuning, experimentation, and running ablation studies.

This guide explains:

- What **Sweeps** and **Multirun** are
- How to use them via the command line interface (CLI)
- How to define sweep parameters in config files
- How to programmatically launch sweeps from Python

---

## What Are Sweeps?

A **Sweep** means running your program multiple times, each time with a different combination of parameter values. Sweeps help explore a parameter space to find optimal or interesting configurations.

---

## What Is Multirun?

**Multirun** is Hydra’s feature for running multiple jobs automatically by combining parameter values. It executes every possible combination of the specified parameters (a Cartesian product) and isolates the results in separate output folders.

---

## Using Sweeps and Multirun via CLI

You can easily launch a sweep from the CLI with the `-m` or `--multirun` flag.

### Example:

```bash
python train.py -m training.lr=0.001,0.01,0.1 training.batch_size=16,32

## Defining Sweep Parameters in YAML Config Files

Hydra does **not** automatically trigger sweeps just by defining parameters in YAML files. However, you can specify the parameters to sweep over as lists in your config files to describe your search space.

### Example `config.yaml`:

```yaml
sweep:
  lr: [0.001, 0.01, 0.1]
  batch_size: [16, 32]

## Running Sweeps Programmatically Using Config Files

Here is an example Python snippet that reads the sweep parameters from your config and launches a run for each combination:

```python
from hydra import initialize, compose
import subprocess
import itertools

initialize(config_path="conf")
cfg = compose(config_name="config")

lrs = cfg.sweep.lr
batch_sizes = cfg.sweep.batch_size

for lr, batch_size in itertools.product(lrs, batch_sizes):
    overrides = [f"training.lr={lr}", f"training.batch_size={batch_size}"]
    subprocess.run(["python", "train.py"] + overrides)


