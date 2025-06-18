To see how we evaluate a model, please refer to [ev_pi0_bridge_simpler.sh](../slurms/eval_scripts/simpler/ev_pi0_bridge_simpler.sh).

Similar to training scripts, eval scripts are all designed to be run on a SLURM cluster, so here we will break down the scripts so you can reuse them on your cluster or local machine without SLURM.


## SLURM Script Breakdown
Eval scripts are, in general, much more complicated. Maniskill2, which SimplerEnv is based on, does not natively support GPU parallelism for simulation, so you can only evaluate one scene at a time, resulting in a slow evaluation. So we had to implement these gimmicks to parallelize the evaluation process. 

We have tried Maniskill3-based SimplerEnv, which natively supports GPU parallelism, but we found two issues: 
1. Discrepancy in metrics (task success rate, etc) between Maniskill2 and Maniskill3 is pretty significant. 
2. Maniskill3 has some unresolved memory leak issues

So, for now, we will stick with Maniskill2.

### Directives
#!/bin/bash
```bash
#SBATCH --job-name=ev_pi0_bridge_simpler
#SBATCH --output=log/slurm/eval/simpler/%x_%j.out
#SBATCH --error=log/slurm/eval/simpler/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint="a100|h100"
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --account=pr_109_tandon_advanced

trap "echo 'Ctrl+C received, killing server...'; kill $SERVER_PID; exit 1" SIGINT
```
The shebang and directives are similar to the training scripts. Note that we use significantly fewer compute resources. 

The `trap` command is a relic for debugging. Feel free to completely remove it if you don't intend to run the script interactively.

Similar to training, **when doing local training**, you can remove all of these except the shebang.

### Config Files and Random Seed
```bash
CONFIG_NAMES=("pi0_finetune_bridge_ev.yaml")
SEEDS=(42 7 314)
```
This section defines which config files and random seeds to use. As you might have noticed, in theory, you can put multiple config files and the script will run them sequentially (e.g., `CONFIG_NAMES=("config1.yaml" "config2.yaml")`). However, we only use one config file in this example.

Random seeds here will affect the random seed used in all RNG-based operations in `torch`, `numpy`, etc that we can think of. This is for reproducibility.

### Idle Port Finding
```bash
# Function to check if a port is available
is_port_in_use() {
    ss -tuln | grep ":$1" > /dev/null
    return $?
}

find_available_port() {
    local port
    for port in $(shuf -i 10000-65500 -n 200); do
        if ! ss -tuln | grep ":$port" > /dev/null; then
            echo $port
            return 0
        fi
    done
    # Fallback to a default port if no random port is found (unlikely)
    echo 5000
    return 1
}
```
Our server-client architecture for inference uses `websocket` under the hood. To avoid race conditions between parallel evaluations, we need to find available ports for each client-server pair. The `is_port_in_use` function is a relic and can be removed. `find_available_port` is the function that's actually being used.

**Even when running locally**, it's recommended to keep this section.

### Environment Variables
```bash
# set all the paths to environment variables
source ./set_path.sh
```
Set the environment variables for paths. Same as the training script.

### Iterate Over Random Seeds and Config Files
```bash
for SEED in "${SEEDS[@]}"; do
    echo "Running with seed $SEED"

    for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
        echo "Running with config $CONFIG_NAME"
```
We iterate over the random seeds and config files defined earlier. These are evaluated sequentially. No parallelism happening.

### Iterate Over Checkpoint Steps
```bash
        STEP_COUNTS=( $(python3 - <<EOF
import yaml, os
from yaml.loader import SafeLoader

# handle !include by loading the referenced file
def include_constructor(loader, node):
    base = os.path.dirname(os.path.abspath("config/experiment/simpler/${CONFIG_NAME}"))
    rel = loader.construct_scalar(node)
    with open(os.path.join(base, rel)) as f:
        return yaml.load(f, Loader=SafeLoader)

SafeLoader.add_constructor('!include', include_constructor)

cfg_path = "config/experiment/simpler/${CONFIG_NAME}"
with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=SafeLoader)

for s in cfg["eval_cfg"]["pretrained_model_gradient_step_cnt"]:
    print(s)
EOF
) )
```
