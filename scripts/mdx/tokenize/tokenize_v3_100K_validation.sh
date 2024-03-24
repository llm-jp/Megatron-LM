#!/bin/bash

set -e

cd /model/llmjp0/Megatron-LM/
source .env/bin/activate

MODEL_PATH="/model/llmjp0/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0a2.model"
OUTPUT_TOKEN_INFO="/model/llmjp0/Megatron-LM//scripts/mdx/tokenize/validation-llm-jp-tokenizer-100k.ver3.0a2.Llama2Tokenizer.csv"

MERGE_DIR="/data/llm-jp-corpus/v1.0.1/merge/val/"

OUTPUT_DIR_VAL="/data/llm-jp-corpus/v1.0.1/merge/binarized/ver3.0/100k"
mkdir -p "$OUTPUT_DIR_VAL"
for file in "$MERGE_DIR"/*.jsonl; 
do
      # ファイル名のみを取得
        base_name=$(basename "$file")

        # 出力ディレクトリのパスと結合
        output_path="$OUTPUT_DIR_VAL/$base_name"
  python tools/preprocess_data.py \
    --input "$file" \
    --output-result-total-token-info $OUTPUT_TOKEN_INFO \
    --output-prefix "$output_path" \
    --tokenizer-model $MODEL_PATH \
    --tokenizer-type Llama2Tokenizer \
    --workers 256 \
    --append-eod
done

