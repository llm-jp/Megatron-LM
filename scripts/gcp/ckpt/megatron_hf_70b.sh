#!/bin/bash
#SBATCH --job-name=ckpt-convert
#SBATCH --time=8:00:00
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/megatron-hf/%x-%j.out
#SBATCH --error=outputs/megatron-hf/%x-%j.out

set -e

# module load
module load cuda/12.1
module load cudnn/8.9.7
module load hpcx/2.17.1

# open file limit
ulimit -n 65536 1048576

# python virtualenv
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=8

# model config
MEGATRON_CHECKPOINT_DIR=/home/ext_kazuki_fujii_turing_motors_c/checkpoints/hf-to-megatron/Llama-2-70b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
HF_CHECKPOINT_DIR=/home/ext_kazuki_fujii_turing_motors_c/checkpoints/megatron-to-hf/Llama-2-70b-hf

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL_DIR=/home/ext_kazuki_fujii_turing_motors_c/hf-checkpoints/Llama-2-70b-hf/

# convert
python tools/checkpoint/util.py \
  --model-type GPT \
  --loader megatron_mcore \
  --saver llama2_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --true-vocab-size 32000 \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --megatron-path /home/ext_kazuki_fujii_turing_motors_c/Megatron-LM
