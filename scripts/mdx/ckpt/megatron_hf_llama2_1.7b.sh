#!/bin/bash

# python virtualenv
cd /model/llmjp0/nii-geniac-megatron/Megatron-LM
source .env/bin/activate
# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1

# Tokenizer config
TOKENIZER_MODEL_DIR="/model/llmjp0/nii-geniac-megatron/Megatron-LM/llama2_1.7B_tokenizer"
# Model config
MEGATRON_CHECKPOINT_DIR="/model/checkpoints_llama2_1.7B/tp1-pp1-cp1"

# Iterations
for ITERATION in $(seq 1009579 100000 1009579); do

  FORMATTED_ITERATION=$(printf "%07d" $ITERATION)
  echo $FORMATTED_ITERATION
  HF_CHECKPOINT_DIR="/model/checkpoints_llama2_1.7B_hf/tp1-pp1-cp1/iter_${FORMATTED_ITERATION}"

  mkdir -p "${HF_CHECKPOINT_DIR}"
  echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

  # Convert
  python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader mcore \
    --saver llama2_hf \
    --load-dir "${MEGATRON_CHECKPOINT_DIR}" \
    --save-dir "${HF_CHECKPOINT_DIR}" \
    --true-vocab-size 99574 \
    --hf-tokenizer-path "${TOKENIZER_MODEL_DIR}" \
    --save-dtype bfloat16 \
    --loader-transformer-impl "transformer_engine" \
    --megatron-path "/model/llmjp0/nii-geniac-megatron/Megatron-LM"
done

echo "Done"