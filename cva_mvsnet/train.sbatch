#!/bin/bash -l

### Calling convention ###
#
# Call like sbatch train.sbatch configs/your_config.yaml [TRAIN.DEVICE slurm-ddp TRAIN.BATCH_SIZE 2]
#
# - The config is mandatory
# - Supplying config values for overwrite in the end is optional
#   (the [] show optionality and should not be used in the call)

# SLURM SUBMIT SCRIPT
# Rules:
#   Set --nodes to the number of nodes
#   Set --gres:gpu:X and --ntasks-per-node=X to the same number
#   Set --mem to about 10*X (e.g. 20G for two GPUs per node)
#   This will give nodes*X total GPUs

#SBATCH --nodes=2
#SBATCH --gres=gpu:2,VRAM:24G
#SBATCH --exclude=node14
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/usr/wiss/%u/slurm/logs/slurm-%j.out

export EXP_DIR="/storage/user/koestlel/dr_experiments/slurm/$SLURM_JOB_ID"
echo "Master Node ($(hostname)) is up. Writing to: $EXP_DIR."

# -------------------------
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Handle Arguments
config=$1
shift 1

srun python train.py --config $config $EXP_DIR TRAIN.DEVICE slurm-ddp $@
