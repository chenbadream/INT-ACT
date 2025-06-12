 <b>This README is still being actively worked on. So if you find something missing, please open an issue and we will take care of it as soon as possible.</b>

:new: [2025-6-12] *Made Public*.

# INT-ACT
### [[Page](https://ai4ce.github.io/INT-ACT/)] | [[Paper](http://arxiv.org/abs/2506.09930)]
This is the official implementation of [From Intention to Execution: Probing the Generalization Boundaries of Vision-Language-Action Models](https://ai4ce.github.io/INT-ACT/)

## Table of Contents
- [Table of Contents](#table-of-contents)
- [TODO](#todo)
- [Installation](#installation)
    - [Install Inference Environment](#install-inference-environment)
        - [Install Inference Client (Simulator) Environment](#install-inference-client-simulator-environment)
        - [(Octo and Magma) Install Inference Server (Policy) Environment](#octo-and-magma-install-inference-server-policy-environment)
- [Acquire Data for Training/Fine-tuning](#acquire-data-for-trainingfine-tuning)
- [Train and Fine-tune](#train-and-fine-tune)
- [Evaluate/Benchmark](#evaluatebenchmark)
- [How to Set ENV Variables](#how-to-set-env-variables)

## TODO
- [ ] Add more complete documentation for training and evaluation. Currently the code is all there but documentation is sparse.

- [ ] Release all relevant model checkpoints on HF

## Installation
Install this codebase by first cloning it.
```
git clone --recurse-submodules https://github.com/ai4ce/INT-ACT.git
cd INT-ACT
```
> [!IMPORTANT]
> See the [How to Set ENV Variables](#how-to-set-env-variables) section for setting up the environment variables.

> [!NOTE]
> This codebase relies on [uv](https://docs.astral.sh/uv/) to manage the virtual environments. It's not strictly required, but the authors can only provide support for this environment management system.

Now simply run 
```
uv sync
```

> [!IMPORTANT]
> This only installed the dependency for training and inference server. Full-scale inference requires installing the inference client (simulator) dependencies.

Inference under different environments, such as `Simpler`, `Simpler-ManiSkill3`, `Libero`, or real world requires installing their own dependency in a separate environment. 

We do this to allow a server-client architecture that can separate the very different compute demands of doing training vs. running experiments on a robot

### Install Inference Environment 
#### Install Inference Client (Simulator) Environment
This example will use Simpler as an example.

1. Create a separate virtual environment for this simulator.
```
cd src/experiments/envs/simpler
```
```
uv venv --python=3.10 # The version can change to accommodate your simulator's need.
```
2. Activate the inference virtual environment. This is important because we don't want to install the simulator dependencies in the training environment.
```
source .venv/bin/activate
```
3. Install the simulator dependencies using `pyproject.toml`
```
uv pip install -r pyproject.toml
```
#### (Octo and Magma) Install Inference Server (Policy) Environment
Octo and Magma both requires specialized policy environments due to conflicting dependencies.
This example will use Octo as an example.
1. Create a separate virtual environment for this policy.
```
cd src/experiments/policies/octo_policy_server
```
```
uv venv --python=3.10 # The version can change to accommodate your simulator's need.
```
2. Activate the inference virtual environment. This is important because we don't want to install the policy dependencies in the training environment.
```
source .venv/bin/activate
```
3. Install the policy dependencies using `pyproject.toml`
```
uv pip install -r pyproject.toml
```

## Acquire Data for Training/Fine-tuning
For now, we refer you to Allen Ren's [README](https://github.com/allenzren/open-pi-zero)

## Train and Fine-tune
1. See `slurms/train_scripts/pi0_finetune_bridge.sh` for how to train
2. See `slurms/train_scripts/pi0_finetune_bridge.sh` for how to finetune

## Evaluate/Benchmark
3. See `slurms/eval_scripts/ev_pi0_bridge_simpler.sh` for how to eval

4. `config` is for all kinds of configuration
    - `config/models` contains model configuration
    - `config/train` contains training pipeline configuration
    - `config/experiment` contains eval pipeline configuration

5. `src` contains important source code
    - `src/experiments` contains simulator adapters and `evaluator.py`
    - `src/agent` contains `trainer.py`. Also contains `run.py`, which serves as the entry point to both `trainer.py` and `evaluator.py`

## How to Set ENV Variables
1. create a `set_path.sh` file in the project's root directory
2. Fill out the following variables
```
#!/bin/bash
# used to sync the path on HPC with data from collaborators and model from baseline directory
# to avoid redundant data download
# training dataset
export VLA_DATA_DIR=

# logging for trained models, logs, etc
export VLA_LOG_DIR=

# WandB
export VLA_WANDB_ENTITY=

# HF cache. TRANSFORMERS_CACHE is deprecated, but still used by some libraries and they themselves a bit confused tbh
export TRANSFORMERS_CACHE=
export HF_HOME=

# SIMPLER
export MS2_REAL2SIM_ASSET_DIR=
export MS_ASSET_DIR=
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# uv (This is optional if you don't mind uv using your home directory, which may not be the case for HPC)
export UV_CACHE_DIR=
export UV_PYTHON_INSTALL_DIR=

# Singularity (This is obviously optional if you don't use singularity)
export OVERLAY_EXT3=
```
