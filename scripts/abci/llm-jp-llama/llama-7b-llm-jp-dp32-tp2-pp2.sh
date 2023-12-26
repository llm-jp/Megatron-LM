#!/bin/bash
#$ -l rt_AF=16
#$ -l h_rt=20:00:00:00
#$ -j y
#$ -o outputs/llm-jp/llama-7b/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# model config
# llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008 # intermediate size (HuggingFace)
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=2   # fixed
PIPELINE_PARALLEL_SIZE=2 # num layers 32: Llama-2 7B
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=119000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
# 118741 iteration = 498B Token -> 119000 iteration

LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/bb/llm/gaf51275/llm-jp/llm-ja-tokenizer/models/ver2/code10K_en20K_ja30K.ver2.2.model
CHECKPOINT_SAVE_DIR="/groups/gaf51275/llm-jp/checkpoints/megatron-lm/7b/llama/llm-jp-v2-corpus/code10K_en20K_ja_30K/context_4096/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# ja wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1489457253 /bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train/ja_wiki/ja_wiki_merge_1_text_document"
# en wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4983898399 /bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train/en_wiki/en_wiki_merge_1_text_document"
# code stack
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8967214774 /bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train/code_stack/code_stack_merge_1_text_document"
# en pile
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17716652494 /bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train/en_pile/en_pile_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17728398911 /bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train/en_pile/en_pile_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17862741217 /bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train/en_pile/en_pile_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17854181202 /bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train/en_pile/en_pile_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17779824310 /bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train/en_pile/en_pile_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17847796716 /bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train/en_pile/en_pile_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8938950206 /bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train/en_pile/en_pile_merge_7_text_document"
# ja cc version2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7942003640 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7136399765 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6779491520 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6438436876 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6400562450 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6350038289 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6319235933 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6172201817 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-21.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6083986223 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5930110667 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5930079046 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5805738111 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5712847332 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5602872845 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-40.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5552132181 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5468192297 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-27.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5248065910 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-29.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5189434471 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-49.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5175829800 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5013839835 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4979044031 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2023-06.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4973927748 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4868716043 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4824673448 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-35.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4776082234 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-49.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4756152936 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-33.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4744751620 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4743990123 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4731899875 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-18.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4695769195 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4670867081 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-10.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4651625740 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4584488704 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-24.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4565377873 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2013-2016.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4510097174 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4503278674 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-16.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4461037639 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4439473560 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4424502587 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-40.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4422439594 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4389130745 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4252352170 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4218017290 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4146455517 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-10.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4134086047 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2023-14.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4078542996 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-21.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4067120082 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-45.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4066527291 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2023-23.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4063614556 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4013050401 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-31.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3884814750 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3783341879 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-50.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3645015224 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3622409047 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3528499517 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-25.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3445433515 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3441240048 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3138193749 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3089993550 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2455160502 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2443269316 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1519216722 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1097557466 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2201753741 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-30.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2011349205 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-09.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2006115573 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-26.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1941245572 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-17.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1879795204 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-26.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1855055519 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-34.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1835767881 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-51.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1810114172 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-43.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1746643758 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-39.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1656998536 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-22.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1627258236 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-09.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1604100276 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-47.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1596422870 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2013-2016.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1564498767 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-34.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1499142823 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-04.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1439458757 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-13.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1429089727 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-47.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1404535062 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-39.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1382317058 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-18.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1339712100 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-22.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1313189571 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-40.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1301330492 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-30.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1294429787 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-35.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1282016690 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-13.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1280245047 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-26.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1236253747 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-29.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1188769660 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-30.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1183776456 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-05.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1160785398 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-16.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1081325944 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-43.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1050397690 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-17.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1046380437 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-04.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1023695606 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-17.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1007693096 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-10.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1001685071 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-39.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1001190547 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-31.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 991674538.5 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-13.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 949849043.1 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-24.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 947062102.3 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-43.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 917301709 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-34.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 916134172.1 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-05.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 910529445.5 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-47.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 863333286.6 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-21.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 859284834.5 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2019-51.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 841808664.8 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-21.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 812554532.3 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-10.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 801879244.9 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-39.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 795908514.4 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-05.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 788468882.7 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-45.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 785723437.3 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2020-50.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 764940395.2 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-27.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 763007647.7 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2018-22.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 736337911.6 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-49.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 701983135.5 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2023-06.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 690225149.4 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2023-23.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 686844983.1 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-40.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 661303881.8 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-25.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 632748989.5 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-51.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 616862290 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2023-14.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 585446143.9 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2021-49.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 555165489.2 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2022-33.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 549254645.4 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-09.jsonl_fil0.1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 354401584.2 /bb/llm/gaf51275/llm-jp/binarize/llama2-7b-500b/ver2.2/code10K_en20K_ja30K/train/CC-ver2.1/CC-MAIN-2017-04.jsonl_fil0.1_text_document"

# job name
JOB_NAME="llama-7b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# checkpoint load
CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -bind-to none -map-by slot \
  -x PATH \
  python pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 969,30,1 \
  --distributed-backend nccl \
  --init-method-std 0.02 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --save-interval 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --use-rotary-position-embeddings \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-position-embedding \
  --no-masked-softmax-fusion \
  --no-query-key-layer-scaling \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --disable-bias-linear \
  --no-bias-gelu-fusion \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --use-mpi \
  --wandb-name ${JOB_NAME} \
  --wandb-project "megatron-lm-7B-2023-1112" \
  --wandb-entity "llm-jp"
