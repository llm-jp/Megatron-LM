#!/bin/bash

set -e

cd /model/llmjp0/Megatron-LM/
source .env/bin/activate

CODE=20
EN=40
JA=60
MODEL_PATH="/model/llm-jp-tokenizer/models/ver2.2/code${CODE}K_en${EN}K_ja${JA}K.ver2.2.model"
OUTPUT_DIR="/data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code${CODE}K_en${EN}K_ja${JA}K/CC-ver2.1"
OUTPUT_TOKEN_INFO="/model/llmjp0/Megatron-LM//scripts/mdx/tokenize/code${CODE}K_en${EN}K_ja${JA}K.ver2.2_v1.0.1_CC-ver2.1b_except_latest31.csv"
mkdir -p $OUTPUT_DIR

MERGE_DIR="/data/llm-jp-corpus-v2.1-CC/merge"


all_files=$(ls "$MERGE_DIR"/*.jsonl)

latest_files=$(ls "$MERGE_DIR"/*.jsonl | tail -n 31)

for file in $all_files; do
    if ! [[ $latest_files == *"$file"* ]]; then
        base_name=$(basename "$file")

        output_path="$OUTPUT_DIR/$base_name"
        python tools/preprocess_data.py \
            --input "$file" \
            --output-result-total-token-info $OUTPUT_TOKEN_INFO \
            --output-prefix "$output_path" \
            --tokenizer-model $MODEL_PATH \
            --tokenizer-type SentencePieceTokenizer \
            --workers 256 \
            --append-eod

        echo "Processing $file"
    fi
done



# for file in `ls "$MERGE_DIR"/*.jsonl|tail -n 31`

# do
#       # ファイル名のみを取得
#         base_name=$(basename "$file")

#         # 出力ディレクトリのパスと結合
#         output_path="$OUTPUT_DIR/$base_name"
#   python tools/preprocess_data.py \
#     --input "$file" \
#     --output-result-total-token-info $OUTPUT_TOKEN_INFO \
#     --output-prefix "$output_path" \
#     --tokenizer-model $MODEL_PATH \
#     --tokenizer-type SentencePieceTokenizer \
#     --workers 256 \
#     --append-eod
# done

# echo "Done 
