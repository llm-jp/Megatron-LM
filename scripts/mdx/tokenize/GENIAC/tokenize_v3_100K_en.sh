#!/bin/bash

set -e

cd /model/llmjp0/Megatron-LM/
source .env/bin/activate

MODEL_PATH="/model/llmjp0/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0a2.model"
OUTPUT_DIR="/data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en"
OUTPUT_TOKEN_INFO="/model/llmjp0/Megatron-LM/scripts/mdx/tokenize/GENIAC/token_info/2024_0410_en.csv"
mkdir -p $OUTPUT_DIR

INPUT_DIR="/data/llm-jp-corpus/v3.0.0/training_resharded/train/en/"

for file in "$INPUT_DIR"/*.jsonl; 
do
  base_name=$(basename "$file")
  output_path="$OUTPUT_DIR/$base_name"
  python tools/preprocess_data.py \
    --input "$file" \
    --output-result-total-token-info $OUTPUT_TOKEN_INFO \
    --output-prefix "$output_path" \
    --tokenizer-model $MODEL_PATH \
    --tokenizer-type Llama2Tokenizer \
    --workers 256 \
    --append-eod
done