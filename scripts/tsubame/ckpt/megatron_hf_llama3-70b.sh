#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=5:00:00
#$ -o outputs/convert/megatron_hf/$JOB_ID
#$ -e outputs/convert/megatron_hf/$JOB_ID
#$ -p -5

# Load modules
module use ~/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.1/2.18.3
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# swich virtual env
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=8

ITERATION=2000
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_CHECKPOINT_DIR=/gs/bs/tgh-NII-LLM/checkpoints/Llama-3-70B/swallow-ja_8-en_1_code_1/tp4-pp8-ct1-LR1.0e-5-MINLR1.0E-6-WD0.05
HF_CHECKPOINT_DIR=/gs/bs/tgh-NII-LLM/checkpoints/megatron-to-hf/Llama-3-70b-hf/LR1.0e-5-MINLR1.0E-6-WD0.05/iter_${FORMATTED_ITERATION}

mkdir -p ${HF_CHECKPOINT_DIR}

echo $ITERATION >"${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/gs/bs/tga-bayes-crest/fujii/hf-checkpoints/Meta-Llama-3-70B

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama3_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl transformer_engine \
  --megatron-path /gs/bs/tga-bayes-crest/fujii/Megatron-LM
