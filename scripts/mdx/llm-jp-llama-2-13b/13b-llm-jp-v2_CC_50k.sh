#!/bin/bash

# python virtualenv
cd /model/llmjp0/Megatron-LM
source .env/bin/activate

# distributed settings
export MASTER_ADDR=10.130.184.10
export MASTER_PORT=12803

echo "MASTER_ADDR=${MASTER_ADDR}"

NODE_TYPE="a100"
export NUM_GPU_PER_NODE=8

HOSTFILE_NAME=/model/llmjp0/Megatron-LM/hostfile/16node


NUM_NODES=16
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# model config
# llama-2-13b: https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
HIDDEN_SIZE=5120
FFN_HIDDEN_SIZE=13824 # intermediate size (HuggingFace)
NUM_LAYERS=40
NUM_HEADS=40
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=2   # fixed
PIPELINE_PARALLEL_SIZE=4 # num layers 40: Llama-2 13B
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=63500 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
# 今回は約270B Tokens #実際の数値は250B Tokens

LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/model/taishi/llm-jp-tokenizer/models/ver2.2/code10K_en20K_ja30K.ver2.2.model
CHECKPOINT_SAVE_DIR=/data/checkpoints_13b/CC_v2_code10K_en20K_ja30K_ver2.2

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code10K_en20K_ja30K

TRAIN_DATA_PATH=""
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1489457253 ${DATASET_DIR}/ja_wiki/ja_wiki_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4983898399 ${DATASET_DIR}/en_wiki/en_wiki_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8967214774 ${DATASET_DIR}/code_stack/code_stack_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17716652494 ${DATASET_DIR}/en_pile/en_pile_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17728398911 ${DATASET_DIR}/en_pile/en_pile_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17862741217 ${DATASET_DIR}/en_pile/en_pile_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17854181202 ${DATASET_DIR}/en_pile/en_pile_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17779824310 ${DATASET_DIR}/en_pile/en_pile_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17847796716 ${DATASET_DIR}/en_pile/en_pile_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8938950206 ${DATASET_DIR}/en_pile/en_pile_merge_7_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 501481782.2 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 694140883.6 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1574242983 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2290858943 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2020643983 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2536808529 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1665433280 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2709495130 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2005424934 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2145530214 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1121780228 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2017-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1572326958 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2924462330 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1433862957 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1411839945 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1116347061 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3260671613 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3628757731 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2610240420 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2652682875 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2887303668 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2779816409 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3097597707 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2018-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2709509577 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2901377483 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2272622827 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2162038578 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-18.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2167562697 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2038279703 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1942929834 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2204427480 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-35.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1856694281 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2060695352 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1774999382 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1655104358 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2019-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2167910631 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2020-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2134152262 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2020-10.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2057579931 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2020-16.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2094685369 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2020-24.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2397878496 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2020-29.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2028426922 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2020-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2559992298 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2020-40.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1858295980 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2020-45.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1728635709 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2020-50.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2224550854 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2021-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1894544903 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2021-10.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2125360757 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2021-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1863515191 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2021-21.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1612196428 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2021-25.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1833591160 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2021-31.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1927241984 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2021-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2364873305 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2021-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2371089376 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2021-49.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2941767424 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2022-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2820122739 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2022-21.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2498455802 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2022-27.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2173119973 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2022-33.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2021586579 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2022-40.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2182225810 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2022-49.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2274960493 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2023-06.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1888893204 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2023-14.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1858025130 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code10K_en20K_ja30K/CC-ver2.1/CC-MAIN-2023-23.jsonl_text_document"

VALID_DATA_PATH=""

VALID_DATA_PATH="${VALID_DATA_PATH} 77810430 ${DATASET_DIR}/val/code_stack_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 37133061 ${DATASET_DIR}/val/en_pile_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 1011609 ${DATASET_DIR}/val/en_wiki_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 147265562 ${DATASET_DIR}/val/ja_cc_validation_0_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 1097003 ${DATASET_DIR}/val/ja_wiki_validation_0_text_document"

# job name
JOB_NAME="13b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# --norm-epsilon 1e-5 : conifg.json (RMS norm)

# # checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
fi

export NCCL_DEBUG=INFO
# run
/usr/local/openmpi-4.1.5/bin/mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=106 \
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
  --use-checkpoint-args \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --train-data-path ${TRAIN_DATA_PATH} \
  --valid-data-path ${VALID_DATA_PATH} \
  --test-data-path ${VALID_DATA_PATH} \
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
  --group-query-attention \
  --num-query-groups 40 \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --use-mpi \
  --wandb-name ${JOB_NAME} \
  --wandb-project "megatron-lm-13B-2023-1225" \
  --wandb-entity "llm-jp"

