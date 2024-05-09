#!/bin/sh
#$ -cwd
#$ -l node_f=2
#$ -l h_rt=0:30:00
#$ -o outputs/Llama-2-13b/$JOB_ID
#$ -e outputs/Llama-2-13b/$JOB_ID
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
# llama-2-13b: https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
HIDDEN_SIZE=5120
FFN_HIDDEN_SIZE=13824 # intermediate size (HuggingFace)
NUM_LAYERS=40
NUM_HEADS=40
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=2   # fixed
PIPELINE_PARALLEL_SIZE=2 # num layers 40: Llama-2 13B
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=500679
LR_DECAY_ITERS=452995

LR=3e-4
MIN_LR=3E-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/gs/bs/tge-gc24sp01/hf-checkpoints/Mixtral-8x7B-Instruct-v0.1-VE/tokenizer.model
CHECKPOINT_SAVE_DIR=/gs/bs/tge-gc24sp01/checkpoints/Llama-2-13b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}

mkdir -p ${CHECKPOINT_SAVE_DIR}

DATA_PATH=""

# ja okazaki lab cc
DATA_PATH="${DATA_PATH} 7417203315 /gs/bs/tge-gc24sp01/datasets/mistral_original_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_0_text_document"

# job name
JOB_NAME="llama-2-13b-base-okazaki-lab-cc-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

# --norm-epsilon 1e-5 : conifg.json (RMS norm)

CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"

export NCCL_DEBUG=INFO

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
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${DATA_PATH} \
  --split 969,30,1 \
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
  --save-interval 100 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --position-embedding-type rope \
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
  --use-embedding-scaling \
  --log-throughput \
  --wandb-name ${JOB_NAME} \
  --wandb-project "TSUBAME-FLOPS" \
  --wandb-entity "okoge"
