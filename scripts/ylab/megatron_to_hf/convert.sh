#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH --job-name=megatron-hf-convert
#SBATCH --time=6:00:00
#SBATCH --output outputs/checkpoint-convert/%j.out
#SBATCH --error errors/checkpoint-convertk/%j.err

# python virtualenv
cd /home/kazuki/llama/Megatron-LM
source .env/bin/activate

# TP > 1, PP > 1の場合は、TP=1, PP=1になるように scripts/abci/change_tp_pp.sh を実行してからconvertしてください
BASE_TENSOR_PARALLEL_SIZE=1  # fixed
BASE_PIPELINE_PARALLEL_SIZE=1 # fixed

SAVE_DIR=/mnt/nfs/Users/kazuki/checkpoints/llama/huggingface-checkpoint/Llama-2-70b-megatron/tp8-pp8/
mkdir -p ${SAVE_DIR}

python scripts/abci/megatron_to_hf/llama_checkpoint_conversion.py \
  --convert_checkpoint_from_megatron_to_transformers \
  --load_path /mnt/nfs/Users/kazuki/checkpoints/llama/llama-megatron-convert-checkpoint-hf/Llama-2-70b/tp${BASE_TENSOR_PARALLEL_SIZE}-pp${BASE_PIPELINE_PARALLEL_SIZE} \
  --save_path $SAVE_DIR \
  --target_params_dtype "fp16" \
  --print-checkpoint-structure \
  --megatron-path /home/kazuki/llama/Megatron-LM
