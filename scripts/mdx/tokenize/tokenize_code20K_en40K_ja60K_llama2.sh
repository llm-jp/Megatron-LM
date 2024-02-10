#!/bin/bash

set -e

cd /model/llmjp0/Megatron-LM/
source .env/bin/activate

CODE=20
EN=40
JA=60
MODEL_PATH="/model/llm-jp-tokenizer/models/ver2.2/code${CODE}K_en${EN}K_ja${JA}K.ver2.2.model"
OUTPUT_DIR="/data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code${CODE}K_en${EN}K_ja${JA}K/CC-ver2.1_Llama2Tokenizer"
OUTPUT_TOKEN_INFO="/model/llmjp0/Megatron-LM//scripts/mdx/tokenize/code${CODE}K_en${EN}K_ja${JA}K.ver2.2_v1.0.1_Llama2Tokenizer.csv"
mkdir -p $OUTPUT_DIR

MERGE_DIR="/data/llm-jp-corpus-v2.1-CC/merge"

OUTPUT_DIR=/data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer
for i in {1..7}; do
  file=/data/llm-jp-corpus/v1.0.1/merge/en_pile/en_pile_merge_${i}.jsonl
  base_name=$(basename "$file")
  output_path="$OUTPUT_DIR/$base_name"
  python tools/preprocess_data.py \
    --input "$file" \
    --output-result-total-token-info "$OUTPUT_TOKEN_INFO" \
    --output-prefix "$output_path" \
    --tokenizer-model "$MODEL_PATH" \
    --tokenizer-type Llama2Tokenizer \
    --workers 256 \
    --append-eod
done

python tools/preprocess_data.py \
  --input /data/llm-jp-corpus/v1.0.1/merge/ja_wiki/ja_wiki_merge_1.jsonl \
  --output-result-total-token-info $OUTPUT_TOKEN_INFO \
  --output-prefix /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/ja_wiki_Llama2Tokenizer/ja_wiki_merge_1 \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 256 \
  --append-eod

python tools/preprocess_data.py \
  --input /data/llm-jp-corpus/v1.0.1/merge/en_wiki/en_wiki_merge_1.jsonl \
  --output-result-total-token-info $OUTPUT_TOKEN_INFO \
  --output-prefix /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_wiki_Llama2Tokenizer/ja_wiki_merge_1 \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 256 \
  --append-eod

python tools/preprocess_data.py \
  --input /data/llm-jp-corpus/v1.0.1/merge/code_stack/code_stack_merge_1.jsonl \
  --output-result-total-token-info $OUTPUT_TOKEN_INFO \
  --output-prefix /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/code_stack_Llama2Tokenizer/code_stack_merge_1.jsonl \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 256 \
  --append-eod





echo "Done"