#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=3:00:00
#$ -o outputs/convert/hf_megatron/$JOB_ID
#$ -e outputs/convert/hf_megatron/$JOB_ID
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

# model config
HF_CHECKPOINT_DIR=/gs/bs/tga-bayes-crest/fujii/hf-checkpoints/Meta-Llama-3-70B
MEGATRON_CHECKPOINT_DIR=/gs/bs/tgh-NII-LLM/checkpoints/hf-to-megatron/Llama-3-70b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/gs/bs/tga-bayes-crest/fujii/hf-checkpoints/Meta-Llama-3-70B/tokenizer.json

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama3_hf \
  --saver mcore \
  --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --bf16 \
  --saver-transformer-impl "transformer_engine"
