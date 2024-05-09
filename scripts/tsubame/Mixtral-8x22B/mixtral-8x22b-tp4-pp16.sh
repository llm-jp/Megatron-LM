#!/bin/sh
#$ -cwd
#$ -l node_f=32
#$ -l h_rt=24:00:00
#$ -o outputs/mixtral-8x22b/$JOB_ID
#$ -e outputs/mixtral-8x22b/$JOB_ID
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
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

# model config
# mixtral-8x22B: https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/main/config.json
HIDDEN_SIZE=6144
FFN_HIDDEN_SIZE=16384 # intermediate size (HuggingFace)
NUM_LAYERS=56
NUM_HEADS=48
NUM_KEY_VALUE_HEADS=8
SEQ_LENGTH=4096

NUM_EXPERTS=8
NUM_EXPERT_TOP_K=2

# distributed settings
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=8
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=12500
LR_DECAY_ITERS=12500

LR=1.0E-5
MIN_LR=1.0E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/gs/bs/tga-bayes-crest/fujii/hf-checkpoints/Meta-Llama-3-8B/tokenizer.json
CHECKPOINT_SAVE_DIR=/gs/bs/tgh-NII-LLM/checkpoints/mixtral-8x22B/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# ja wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1691212948 /gs/bs/tga-bayes-crest/Swallow/binarized/Meta-Llama-3/ja_wiki_merged_text_document"

# en wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2000000000 /gs/bs/tga-bayes-crest/Swallow/binarized/Meta-Llama-3/en_wiki_merged_train_text_document"

# job name
JOB_NAME="Mixtral-8x22b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

# checkpoint load
CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -bind-to none \
  -x PATH \
  python pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --group-query-attention \
  --num-query-groups ${NUM_KEY_VALUE_HEADS} \
  --num-experts ${NUM_EXPERTS} \
  --moe-router-load-balancing-type aux_loss \
  --moe-grouped-gemm \
  --moe-router-topk 2 \
  --moe-aux-loss-coeff 0.001 \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 998,1,1 \
  --distributed-backend nccl \
  --init-method-std 0.02 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-decay-iters ${LR_DECAY_ITERS} \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --save-interval 500 \
  --eval-interval 500 \
  --eval-iters 10 \
  --bf16 \
  --use-checkpoint-args \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
  --position-embedding-type rope \
  --rope-theta 1000000.0 \
  --disable-bias-linear \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --attention-softmax-in-fp32 \
  --transformer-impl "transformer_engine" \
  --fp8-format 'hybrid' \
  --use-mpi \
  --use-z-loss \
  --log-throughput \
  --wandb-name ${JOB_NAME} \
  --wandb-project "TSUBAME-MoE" \
  --wandb-entity "llm-jp"
