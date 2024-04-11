#!/bin/bash

# python virtualenv
cd /model/taishi/Megatron-LM
source .env/bin/activate

# distributed settings
export MASTER_ADDR=10.130.184.10
export MASTER_PORT=12802

echo "MASTER_ADDR=${MASTER_ADDR}"

NODE_TYPE="a100"
export NUM_GPU_PER_NODE=8

HOSTFILE_NAME=/model/taishi/Megatron-LM/scripts/mdx/hostfile/16node


NUM_NODES=16
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

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
TRAIN_STEPS=61000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
# 今回は約270B Tokens #実際の数値は250B Tokens

LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/model/llmjp0/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0a2.model 
CHECKPOINT_SAVE_DIR=/model/checkpoints_7b/CC_v2_100k_ver3.0_z_loss_only
CHECKPOINT_SAVE_DIR_NEXT=/data/checkpoints_7b/CC_v2_100k_ver3.0_z_loss_only

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATA_PATH=""
DATA_PATH="${DATA_PATH} 8448584668 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//en_pile_Llama2Tokenizer/en_pile_merge_7.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16870094553 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//en_pile_Llama2Tokenizer/en_pile_merge_6.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16808903238 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//en_pile_Llama2Tokenizer/en_pile_merge_5.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16885376419 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//en_pile_Llama2Tokenizer/en_pile_merge_4.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16883695256 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//en_pile_Llama2Tokenizer/en_pile_merge_3.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16762800174 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//en_pile_Llama2Tokenizer/en_pile_merge_2.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16755181304 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//en_pile_Llama2Tokenizer/en_pile_merge_1.jsonl_text_document"
DATA_PATH="${DATA_PATH} 4650036808 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//en_wiki_Llama2Tokenizer/en_wiki_merge_1_text_document"
DATA_PATH="${DATA_PATH} 1198745775 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//ja_wiki_Llama2Tokenizer//ja_wiki_merge_1_text_document"
DATA_PATH="${DATA_PATH} 8415284499 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//code_stack_Llama2Tokenizer/code_stack_merge_1.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1817987379 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2023-23.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1854579391 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2023-14.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2236931223 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2023-06.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2139626049 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-49.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1971498565 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-40.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2118800235 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-33.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2424947019 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-27.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2729093620 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-21.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2823986952 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-05.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2284903391 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-49.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2295512753 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-43.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1879658267 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-39.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1786658735 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-31.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1571148502 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-25.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1817807219 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-21.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2068292794 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-17.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1842921907 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-10.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2160765130 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-04.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1681720917 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-50.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1806904294 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-45.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2490305289 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-40.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1976539999 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-34.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2336208962 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-29.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2041428312 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-24.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2005536488 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-16.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2081598290 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-10.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2112091680 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-05.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1613196488 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-51.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1728865729 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-47.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2007635793 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-43.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1812253549 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-39.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2148172345 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-35.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1895275683 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-30.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1988949438 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-26.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2115342344 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-22.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2106005827 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-18.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2217968035 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-13.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2823724588 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-09.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2644561970 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-04.jsonl_text_document"
DATA_PATH="${DATA_PATH} 3009618533 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-51.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2712750316 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-47.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2814362188 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-43.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2589264263 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-39.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2544939435 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-34.jsonl_text_document"
DATA_PATH="${DATA_PATH} 3539429525 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-30.jsonl_text_document"
DATA_PATH="${DATA_PATH} 3183669145 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-26.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1087666961 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-22.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1376106180 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-17.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1398634667 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-13.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2854476768 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-09.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1533469901 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-05.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1092521680 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-51.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2095191623 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-47.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1957379949 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-39.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2640646173 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-34.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1622112197 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-30.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2472442102 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-26.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1972825268 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-22.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2235348623 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-17.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1538492140 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-13.jsonl_text_document"
DATA_PATH="${DATA_PATH} 680320256.1 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-09.jsonl_text_document"
DATA_PATH="${DATA_PATH} 490503590.2 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-04.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2038969247 /data/llm-jp-corpus-v2.1-CC/binarized/ver3.0/llm-jp-tokenizer-100k.ver3.0a2//CC-ver2.1_Llama2Tokenizer/CC-MAIN-2013-2016.jsonl_text_document"
# job name
JOB_NAME="7b-z_loss_only-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# --norm-epsilon 1e-5 : conifg.json (RMS norm)

# # checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR} --no-load-rng --no-load-optim"
fi

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=106 \
  -bind-to none -map-by slot \
  -x PATH \
  -x CUDA_LAUNCH_BLOCKING \
  -x NCCL_DEBUG \
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
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --save ${CHECKPOINT_SAVE_DIR_NEXT} \
  --data-path ${DATA_PATH} \
  --split 949,50,1 \
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
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --disable-bias-linear \
  --no-bias-gelu-fusion \
  --group-query-attention \
  --num-query-groups 32 \
  --swiglu \
  --use-flash-attn \
  --use-z-loss \
  --recompute-activations \
  --recompute-granularity "selective" \
  --use-mpi \
  --wandb-name ${JOB_NAME} \
  --wandb-project "megatron-lm-7B-2024-0303-ver3.0" \
  --wandb-entity "llm-jp"
