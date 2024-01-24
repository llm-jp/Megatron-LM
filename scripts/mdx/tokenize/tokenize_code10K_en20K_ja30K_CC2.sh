#!/bin/bash

set -e

cd /model/llmjp0/Megatron-LM/
source .env/bin/activate

CODE=10
EN=20
JA=30
MODEL_PATH="/model/llm-jp-tokenizer/models/ver2.2/code${CODE}K_en${EN}K_ja${JA}K.ver2.2.model"
OUTPUT_DIR="/data/llm-jp-corpus-v2-CC/binarized/ver2.2/code${CODE}K_en${EN}K_ja${JA}K/CC-ver2"
OUTPUT_TOKEN_INFO="/model/llmjp0/Megatron-LM//scripts/mdx/tokenize/code${CODE}K_en${EN}K_ja${JA}K.ver2.2_v1.0.1_CC-ver2.csv"
mkdir -p $OUTPUT_DIR

MERGE_DIR="/data/llm-jp-corpus-v2-CC/filtered_v0/merge_reform"


for file in "$MERGE_DIR"/*.jsonl

do
      # ファイル名のみを取得
        base_name=$(basename "$file")

        # 出力ディレクトリのパスと結合
        output_path="$OUTPUT_DIR/$base_name"
  python tools/preprocess_data.py \
    --input "$file" \
    --output-result-total-token-info $OUTPUT_TOKEN_INFO \
    --output-prefix "$output_path" \
    --tokenizer-model $MODEL_PATH \
    --tokenizer-type SentencePieceTokenizer \
    --workers 256 \
    --append-eod
done

echo "Done 
