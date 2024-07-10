#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=5:00:00
#$ -o outputs/ckpt-convert/$JOB_ID.out
#$ -e outputs/ckpt-convert/$JOB_ID.out
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

set -e

# TP, PP change
CHECKPOINT_DIR=/gs/bs/tgh-NII-LLM/checkpoints/Llama-2-13b/CC_v2_code20K_en40K_ja60K_ver2.2/exp-A
CONVERTED_CHECKPOINT_DIR=/gs/bs/tgh-NII-LLM/checkpoints/Llama-2-13b/CC_v2_code20K_en40K_ja60K_ver2.2/exp-A-tp1-pp1

ITERATION=5000
echo $ITERATION >${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt

TRAINING_TP=2
TRAINING_PP=2

mkdir -p ${CONVERTED_CHECKPOINT_DIR}

python tools/checkpoint/util.py \
  --model-type GPT \
  --loader megatron \
  --saver megatron \
  --megatron-path /gs/bs/tga-NII-LLM/src/Megatron-LM-mdx \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 1 \
  --load-dir ${CHECKPOINT_DIR} \
  --save-dir ${CONVERTED_CHECKPOINT_DIR}

SAVE_DIR=/gs/bs/tgh-NII-LLM/checkpoints/hf-to-megatron/Llama-2-13b/CC_v2_code20K_en40K_ja60K_ver2.2/exp-A/iter-${ITERATION}
mkdir -p ${SAVE_DIR}

python scripts/abci/megatron_to_hf/megatron_to_hf.py \
  --convert_checkpoint_from_megatron_to_transformers \
  --load_path $CONVERTED_CHECKPOINT_DIR \
  --save_path $SAVE_DIR \
  --target_params_dtype "bf16" \
  --print-checkpoint-structure \
  --megatron-path /gs/bs/tga-NII-LLM/src/Megatron-LM-mdx

# tokenizer config copy
cp /gs/bs/tga-NII-LLM/src/llm-jp-tokenizer/hf/ver2.2/code20K_en40K_ja60K.ver2.2_hf_fast.b4/* $SAVE_DIR/

# change special tokens
python /gs/bs/tga-NII-LLM/src/modelwg/convert_special_token_settings.py $SAVE_DIR/
