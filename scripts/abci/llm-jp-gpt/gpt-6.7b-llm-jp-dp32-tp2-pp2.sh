#!/bin/bash
#$ -l rt_AF=16
#$ -l h_rt=10:00:00:00
#$ -j y
#$ -o outputs/llm-jp/gpt-6.7b/
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
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=$((${HIDDEN_SIZE} * 4))  # gpt architecuture
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=2
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=63312

LR=1e-4
MIN_LR=3.3e-6
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/bb/llm/gaf51275/llm-jp/llm-ja-tokenizer/models/ver2/code10K_en20K_ja30K.ver2.2.model
CHECKPOINT_SAVE_DIR="/groups/gaf51275/llm-jp/checkpoints/megatron-lm/7b/llm-jp-v1.0.1/code10K_en20K_ja_30K/context_4096/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train

TRAIN_DATA_PATH=""

# ja wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1489457253 ${DATASET_DIR}/ja_wiki/ja_wiki_merge_1_text_document"
# en wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4983898399 ${DATASET_DIR}/en_wiki/en_wiki_merge_1_text_document"
# code stack
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8967214774 ${DATASET_DIR}/code_stack/code_stack_merge_1_text_document"
# en pile
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17716652494 ${DATASET_DIR}/en_pile/en_pile_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17728398911 ${DATASET_DIR}/en_pile/en_pile_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17862741217 ${DATASET_DIR}/en_pile/en_pile_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17854181202 ${DATASET_DIR}/en_pile/en_pile_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17779824310 ${DATASET_DIR}/en_pile/en_pile_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17847796716 ${DATASET_DIR}/en_pile/en_pile_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8938950206 ${DATASET_DIR}/en_pile/en_pile_merge_7_text_document"
# ja cc
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19540410239 ${DATASET_DIR}/ja_cc/ja_cc_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19559059958 ${DATASET_DIR}/ja_cc/ja_cc_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19547251566 ${DATASET_DIR}/ja_cc/ja_cc_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19550089401 ${DATASET_DIR}/ja_cc/ja_cc_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19553509796 ${DATASET_DIR}/ja_cc/ja_cc_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19566479585 ${DATASET_DIR}/ja_cc/ja_cc_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17060823775 ${DATASET_DIR}/ja_cc/ja_cc_merge_7_text_document"

# validation data
VALIDATION_DATASET_PATH="/bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/val"

VALIDATION_DATA_PATH=""
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 77810430 ${VALIDATION_DATASET_PATH}/code_stack_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 37133061 ${VALIDATION_DATASET_PATH}/en_pile_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 1011609 ${VALIDATION_DATASET_PATH}/en_wiki_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 147265562 ${VALIDATION_DATASET_PATH}/ja_cc_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 1097003 ${VALIDATION_DATASET_PATH}/ja_wiki_validation_0_text_document"


# job name
JOB_NAME="gpt-6.7b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

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
  --train-data-path ${TRAIN_DATA_PATH} \
  --valid-data-path ${VALIDATION_DATA_PATH} \
  --distributed-backend nccl \
  --init-method-std 0.009 \
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
  --normalization "LayerNorm" \
  --untie-embeddings-and-output-weights \
  --use-rotary-position-embeddings \
  --no-position-embedding \
  --no-masked-softmax-fusion \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --use-mpi \
  --wandb-name ${JOB_NAME} \
  --wandb-project "megatron-lm-7B-2023-1112" \
  --wandb-entity "llm-jp"
