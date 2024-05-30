#!/bin/bash

# python virtualenv
cd /model/llmjp0/nii-geniac-megatron/Megatron-LM
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1

# model config
MEGATRON_CHECKPOINT_DIR=/model/checkpoints_1.3b_mcore/CC_v2_code20K_en40K_ja60K_ver2.2/
HF_CHECKPOINT_DIR=/model/checkpoints_1.3b_hf/megatron_to_hf/CC_v2_code20K_en40K_ja60K_ver2.2/

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL_DIR=/model/checkpoints_1.3b_hf/megatron_to_hf/CC_v2_code20K_en40K_ja60K_ver2.2/

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama2_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --true-vocab-size 96867 \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl "transformer_engine" \
  --megatron-path /model/llmjp0/nii-geniac-megatron/Megatron-LM
