#!/bin/bash

set -e

start=5000
end=10000
increment=5000

upload_base_dir=/gs/bs/tgh-NII-LLM/checkpoints/hf-to-megatron/Llama-2-13b/CC_v2_code20K_en40K_ja60K_ver2.2/exp-A

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter-$i

  python scripts/abci/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name llm-jp/llm-jp-geniac-debug-expA-iter$(printf "%07d" $i)
done
