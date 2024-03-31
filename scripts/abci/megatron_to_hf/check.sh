#!/bin/bash
#SBATCH --job-name=ckpt-convert-check
#SBATCH --time=5:00:00
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/check/%x-%j.out
#SBATCH --error=outputs/check/%x-%j.out

set -e

# module load
module load cuda/12.1
module load cudnn/8.9.7
module load hpcx/2.17.1

# open file limit
ulimit -n 65536 1048576

# python virtualenv
source .env/bin/activate

python scripts/abci/megatron_to_hf/check.py \
  --base-hf-model-path /home/ext_kazuki_fujii_turing_motors_c/hf-checkpoints/Llama-2-7b-hf \
  --converted-hf-model-path /home/ext_kazuki_fujii_turing_motors_c/checkpoints/megatron-to-hf/Llama-2-7b-hf
