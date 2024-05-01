#!/bin/bash

set -e

start=1500
end=1500
increment=500

upload_base_dir=/gs/bs/tgh-NII-LLM/checkpoints/megatron-to-hf/Llama-3-8b-hf/LR1.0E-4-MINLR1.0E-5-WD0.1

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/abci/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Llama3-8b-LR1.0E-4-MINLR1.0E-5-WD0.1-iter$(printf "%07d" $i)
done
