#!/bin/bash

# python virtualenv
cd /model/llmjp0/nii-geniac-megatron/Megatron-LM
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1

# model config
MEGATRON_CHECKPOINT_DIR=/model/checkpoints_1.3b_mcore/CC_v2_code20K_en40K_ja60K_ver2.2/

HF_CHECKPOINT_DIR=/model/checkpoints_1.3b_hf/CC_v2_code20K_en40K_ja60K_ver2.2/
MEGATRON_CHECKPOINT_DIR=/model/checkpoints_1.3b_mcore/CC_v2_code20K_en40K_ja60K_ver2.2/

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/model/llmjp0/llm-jp-tokenizer/models/ver2.2/code20K_en40K_ja60K.ver2.2.model

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama2_hf \
  --saver mcore \
  --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --bf16 \
  --saver-transformer-impl "transformer_engine"
