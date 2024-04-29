#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --time=3:00:00
#SBATCH --partition=a3
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/tokenize/%x-%j.out
#SBATCH --error=outputs/tokenize/%x-%j.out

# module load
module load cuda/12.1
module load cudnn/8.9.7
module load hpcx/2.17.1

# python virtualenv
source .env/bin/activate

DATASET_DIR=/home/ext_kazuki_fujii_rio_gsic_titech/datasets/samples
OUTPUT_DIR=/home/ext_kazuki_fujii_rio_gsic_titech/datasets/samples

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wikipedia
python tools/preprocess_data.py \
  --input ${DATASET_DIR}/ja_wiki.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model /home/ext_kazuki_fujii_rio_gsic_titech/hf_checkpoints/Meta-Llama-3-8B/tokenizer.jsonl \
  --append-eod \
  --workers 64
