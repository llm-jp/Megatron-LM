#!/bin/bash

set -e

cd /home/taishi/Megatron-LM
source .env/bin/activate

CODE=10
EN=20
JA=30
MODEL_PATH="/model/taishi/llm-jp-tokenizer/models/ver2.2/code${CODE}K_en${EN}K_ja${JA}K.ver2.2.model"
OUTPUT_DIR="/data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code${CODE}K_en${EN}K_ja${JA}K/val"
OUTPUT_TOKEN_INFO="/home/taishi/Megatron-LM/scripts/mdx/tokenize/code${CODE}K_en${EN}K_ja${JA}K.ver2.2_v1.0.1_okazaki_cc.csv"
mkdir -p $OUTPUT_DIR

SPLITS_DIR="/data/taishi/datasets/cc"
OUTPUT_DIR="/data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code${CODE}K_en${EN}K_ja${JA}K/okazaki_cc"
mkdir -p $OUTPUT_DIR

for i in {4..4}
do
  SPLIT_FILE="${SPLITS_DIR}/split_${i}.jsonl"
  OUTPUT_PREFIX="${OUTPUT_DIR}/okazaki_cc_merge_${i}"
  python tools/preprocess_data.py \
    --input "$SPLIT_FILE" \
    --output-result-total-token-info $OUTPUT_TOKEN_INFO \
    --output-prefix "$OUTPUT_PREFIX" \
    --tokenizer-model $MODEL_PATH \
    --tokenizer-type SentencePieceTokenizer \
    --workers 128 \
    --append-eod
done

echo "Done $SPLITS_DIR"