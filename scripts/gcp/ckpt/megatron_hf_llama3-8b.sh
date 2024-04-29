#!/bin/bash
#SBATCH --job-name=ckpt-convert
#SBATCH --time=5:00:00
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
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=2

# model config
MEGATRON_CHECKPOINT_DIR=/home/ext_kazuki_fujii_rio_gsic_titech/checkpoints/hf-to-megatron/Llama-3-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
HF_CHECKPOINT_DIR=/home/ext_kazuki_fujii_rio_gsic_titech/checkpoints/megatron-to-hf/Llama-3-8b-hf

mkdir -p ${HF_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL_DIR=/home/ext_kazuki_fujii_rio_gsic_titech/hf_checkpoints/Meta-Llama-3-8B

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama3_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl "transformer_engine" \
  --megatron-path /home/ext_kazuki_fujii_rio_gsic_titech/src/Megatron-LM
