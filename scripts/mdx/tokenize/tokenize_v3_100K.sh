#!/bin/bash

set -e

cd /model/llmjp0/Megatron-LM/
source .env/bin/activate

MODEL_PATH="/model/llmjp0/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0a2.model"
OUTPUT_DIR="/data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2/"
OUTPUT_TOKEN_INFO="/model/llmjp0/Megatron-LM//scripts/mdx/tokenize/llm-jp-tokenizer-100k.ver3.0a2.Llama2Tokenizer.csv"
mkdir -p $OUTPUT_DIR

MERGE_DIR="/data/llm-jp-corpus-v2.1-CC/merge"


OUTPUT_DIR_CC="${OUTPUT_DIR}/CC-ver2.1_Llama2Tokenizer"
mkdir -p "$OUTPUT_DIR_CC"
for file in "$MERGE_DIR"/*.jsonl; 
do
      # ファイル名のみを取得
        base_name=$(basename "$file")

        # 出力ディレクトリのパスと結合
        output_path="$OUTPUT_DIR_CC/$base_name"
  python tools/preprocess_data.py \
    --input "$file" \
    --output-result-total-token-info $OUTPUT_TOKEN_INFO \
    --output-prefix "$output_path" \
    --tokenizer-model $MODEL_PATH \
    --tokenizer-type Llama2Tokenizer \
    --workers 256 \
    --append-eod
done

OUTPUT_DIR_PILE="${OUTPUT_DIR}/en_pile_Llama2Tokenizer"
mkdir -p "$OUTPUT_DIR_PILE"
for i in {1..7}; do
  file=/data/llm-jp-corpus/v1.0.1/merge/en_pile/en_pile_merge_${i}.jsonl
  base_name=$(basename "$file")
  output_path="${OUTPUT_DIR_PILE}/$base_name"
  python tools/preprocess_data.py \
    --input "$file" \
    --output-result-total-token-info "$OUTPUT_TOKEN_INFO" \
    --output-prefix "$output_path" \
    --tokenizer-model "$MODEL_PATH" \
    --tokenizer-type Llama2Tokenizer \
    --workers 256 \
    --append-eod
done

OUTPUT_DIR_JA_WIKI="${OUTPUT_DIR}/ja_wiki_Llama2Tokenizer/"
mkdir -p "$OUTPUT_DIR_JA_WIKI"
python tools/preprocess_data.py \
  --input /data/llm-jp-corpus/v1.0.1/merge/ja_wiki/ja_wiki_merge_1.jsonl \
  --output-result-total-token-info $OUTPUT_TOKEN_INFO \
  --output-prefix "${OUTPUT_DIR_JA_WIKI}/ja_wiki_merge_1" \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 256 \
  --append-eod

OUTPUT_DIR_EN_WIKI="${OUTPUT_DIR}/en_wiki_Llama2Tokenizer"
mkdir -p "$OUTPUT_DIR_EN_WIKI"
python tools/preprocess_data.py \
  --input /data/llm-jp-corpus/v1.0.1/merge/en_wiki/en_wiki_merge_1.jsonl \
  --output-result-total-token-info $OUTPUT_TOKEN_INFO \
  --output-prefix "${OUTPUT_DIR_EN_WIKI}/en_wiki_merge_1" \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 256 \
  --append-eod

OUTPUT_DIR_CODE="${OUTPUT_DIR}/code_stack_Llama2Tokenizer"
mkdir -p "$OUTPUT_DIR_CODE"
python tools/preprocess_data.py \
  --input /data/llm-jp-corpus/v1.0.1/merge/code_stack/code_stack_merge_1.jsonl \
  --output-result-total-token-info $OUTPUT_TOKEN_INFO \
  --output-prefix "${OUTPUT_DIR_CODE}/code_stack_merge_1.jsonl" \
  --tokenizer-model $MODEL_PATH \
  --tokenizer-type Llama2Tokenizer \
  --workers 256 \
  --append-eod


echo "Done" 

