#!/bin/bash
#SBATCH --job-name=ckpt-convert
#SBATCH --time=5:00:00
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/hf-megatron/%x-%j.out
#SBATCH --error=outputs/hf-megatron/%x-%j.out

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
HF_CHECKPOINT_DIR=/home/ext_kazuki_fujii_rio_gsic_titech/hf_checkpoints/Meta-Llama-3-8B
MEGATRON_CHECKPOINT_DIR=/home/ext_kazuki_fujii_rio_gsic_titech/checkpoints/hf-to-megatron/Llama-3-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/home/ext_kazuki_fujii_rio_gsic_titech/hf_checkpoints/Meta-Llama-3-8B/tokenizer.json

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama3_hf \
  --saver mcore \
  --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --bf16 \
  --saver-transformer-impl "transformer_engine"
