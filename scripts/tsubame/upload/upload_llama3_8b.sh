#!/bin/bash

set -e

start=1500
end=1500
increment=500

base_dirs=(
  "/gs/bs/tgh-NII-LLM/checkpoints/Llama-3-8b/swallow-ja_8-en_1_code_1/LR1.0e-5-MINLR1.0E-6-WD0.05"
  "/gs/bs/tgh-NII-LLM/checkpoints/Llama-3-8b/swallow-ja_8-en_1_code_1/LR2.5e-5-MINLR2.5E-6-WD0.05"
)

for base_dir in "${base_dirs[@]}"; do
  for ((i = start; i <= end; i += increment)); do
    upload_dir=$base_dir/iter_$(printf "%07d" $i)

    python scripts/abci/upload/upload.py \
      --ckpt-path $upload_dir \
      --repo-name tokyotech-llm/Llama3-8b-$(basename $base_dir)-iter$(printf "%07d" $i)
  done
done
