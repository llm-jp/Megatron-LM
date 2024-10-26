#!/bin/bash
#SBATCH --job-name=0059_v3-8x1.8b-exp1
#SBATCH --partition=gpu-debug
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -eu -o pipefail

module load cuda/12.1
module load /data/cudnn-tmp-install/modulefiles/8.9.4
module load hpcx/2.17.1-gcc-cuda12/hpcx
module load nccl/2.20.5
source scripts/sakura/example/mpi_variables.sh 
source venv/bin/activate

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=$(echo $SLURM_TASKS_PER_NODE | cut -d '(' -f 1)
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS


# open file limit
ulimit -n 65536 1048576

export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1


# model config
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=7168
NUM_LAYERS=24
NUM_HEADS=16
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=2
EXPERT_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE} * ${EXPERT_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024

LR=3e-4
MIN_LR=3e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# total number of iterations
# 2072488058295 (number of tokens) / 4096 (seq len) / 1024 (batch size) = 494119.65806365 -> 494120
LR_WARMUP_STEPS=2000
LR_DECAY_ITERS=492120
TRAIN_STEPS=492120

CACHE_DIR=cache

TOKENIZER_MODEL="llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model"
CHECKPOINT_PATH=checkpoints

mkdir -p ${CHECKPOINT_PATH}

export WANDB_PROJECT="Megatron-LM-8x1.8B"
export WANDB_NAME="${NUM_NODES}nodes_${NUM_GPUS}gpus_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}_ep${EXPERT_PARALLEL_SIZE}_cp_${CONTEXT_PARALLEL_SIZE}_dp${DATA_PARALLEL_SIZE}"

# data config
DATASET_DIR=/home/shared/corpus/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0
DATASET_V3_1_DIR=/home/shared/corpus/llm-jp-corpus/v3.1.0/tokenize/v3.0b1

DATA_PATH=""

# code stack
DATA_PATH="${DATA_PATH} 14486363187 ${DATASET_DIR}/train/code/stack_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 12799385151 ${DATASET_DIR}/train/code/stack_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17282923545 ${DATASET_DIR}/train/code/stack_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 8861329235 ${DATASET_DIR}/train/code/stack_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 6713413649 ${DATASET_DIR}/train/code/stack_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 8976432285 ${DATASET_DIR}/train/code/stack_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17961273649 ${DATASET_DIR}/train/code/stack_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 12016948303 ${DATASET_DIR}/train/code/stack_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14953094719 ${DATASET_DIR}/train/code/stack_0008.jsonl_text_document"

# ja cc 1
DATA_PATH="${DATA_PATH} 23783124862 ${DATASET_DIR}/train/ja/cc-1_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 36378129564 ${DATASET_DIR}/train/ja/cc-1_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 35477545812 ${DATASET_DIR}/train/ja/cc-1_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 35917231868 ${DATASET_DIR}/train/ja/cc-1_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 46203062776 ${DATASET_DIR}/train/ja/cc-1_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 40396278536 ${DATASET_DIR}/train/ja/cc-1_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 33444216206 ${DATASET_DIR}/train/ja/cc-1_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 32375495374 ${DATASET_DIR}/train/ja/cc-1_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 36068919622 ${DATASET_DIR}/train/ja/cc-1_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 26274952324 ${DATASET_DIR}/train/ja/cc-1_0009.jsonl_text_document"
DATA_PATH="${DATA_PATH} 24024422756 ${DATASET_DIR}/train/ja/cc-1_0010.jsonl_text_document"
DATA_PATH="${DATA_PATH} 34590145510 ${DATASET_DIR}/train/ja/cc-1_0011.jsonl_text_document"
DATA_PATH="${DATA_PATH} 29567301906 ${DATASET_DIR}/train/ja/cc-1_0012.jsonl_text_document"
DATA_PATH="${DATA_PATH} 26690562242 ${DATASET_DIR}/train/ja/cc-1_0013.jsonl_text_document"

# ja cc 2
DATA_PATH="${DATA_PATH} 35813749376 ${DATASET_DIR}/train/ja/cc-2_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 40034668924 ${DATASET_DIR}/train/ja/cc-2_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 31191828858 ${DATASET_DIR}/train/ja/cc-2_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 25086109508 ${DATASET_DIR}/train/ja/cc-2_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18979589830 ${DATASET_DIR}/train/ja/cc-2_0004.jsonl_text_document"

# ja cc 3
DATA_PATH="${DATA_PATH} 40987803038 ${DATASET_DIR}/train/ja/cc-3_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 41333549162 ${DATASET_DIR}/train/ja/cc-3_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 29810274406 ${DATASET_DIR}/train/ja/cc-3_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 22787733940 ${DATASET_DIR}/train/ja/cc-3_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15544493906 ${DATASET_DIR}/train/ja/cc-3_0004.jsonl_text_document"

# ja kaken
DATA_PATH="${DATA_PATH} 1826105478 ${DATASET_DIR}/train/ja/kaken_0000.jsonl_text_document"

# ja warp html
DATA_PATH="${DATA_PATH} 1329440698 ${DATASET_DIR}/train/ja/warp-html-01-06_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1397268214 ${DATASET_DIR}/train/ja/warp-html-07-12_0000.jsonl_text_document"

# ja warp pdf
DATA_PATH="${DATA_PATH} 30149711608 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e00_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 30023232706 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e00_0001.jsonl_text_document"

# ja warp pdf 0.2
DATA_PATH="${DATA_PATH} 15396388677 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 13225220331 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 12433511477 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14722870558 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14818300138 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14827819309 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 13394854115 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14369730518 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14027593174 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14719994730 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0009.jsonl_text_document"
DATA_PATH="${DATA_PATH} 9865165774 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0010.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14525215128 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0011.jsonl_text_document"
DATA_PATH="${DATA_PATH} 10835111330 ${DATASET_V3_1_DIR}/train2/ja/warp-pdf-e02_0012.jsonl_text_document"

# ja wiki
DATA_PATH="${DATA_PATH} 2563804308 ${DATASET_DIR}/train/ja/wiki_0000.jsonl_text_document"

# en dolma books
DATA_PATH="${DATA_PATH} 5494262694 ${DATASET_DIR}/train/en/dolma-books_0000.jsonl_text_document"

# en dolma c4
DATA_PATH="${DATA_PATH} 17052861266 ${DATASET_DIR}/train/en/dolma-c4_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17051260422 ${DATASET_DIR}/train/en/dolma-c4_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17056648148 ${DATASET_DIR}/train/en/dolma-c4_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17057773049 ${DATASET_DIR}/train/en/dolma-c4_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17047888113 ${DATASET_DIR}/train/en/dolma-c4_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17046511755 ${DATASET_DIR}/train/en/dolma-c4_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17058086815 ${DATASET_DIR}/train/en/dolma-c4_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17049490900 ${DATASET_DIR}/train/en/dolma-c4_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17051009552 ${DATASET_DIR}/train/en/dolma-c4_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14932405246 ${DATASET_DIR}/train/en/dolma-c4_0009.jsonl_text_document"
DATA_PATH="${DATA_PATH} 13142696712 ${DATASET_DIR}/train/en/dolma-c4_0010.jsonl_text_document"

# en dolma cc
DATA_PATH="${DATA_PATH} 15473522696 ${DATASET_DIR}/train/en/dolma-cc-head_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15767913273 ${DATASET_DIR}/train/en/dolma-cc-head_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16664785078 ${DATASET_DIR}/train/en/dolma-cc-head_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16860035920 ${DATASET_DIR}/train/en/dolma-cc-head_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17197613512 ${DATASET_DIR}/train/en/dolma-cc-head_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16363353173 ${DATASET_DIR}/train/en/dolma-cc-head_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15303692924 ${DATASET_DIR}/train/en/dolma-cc-head_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15766283829 ${DATASET_DIR}/train/en/dolma-cc-head_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 13483997219 ${DATASET_DIR}/train/en/dolma-cc-head_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 12561851173 ${DATASET_DIR}/train/en/dolma-cc-head_0009.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14206017429 ${DATASET_DIR}/train/en/dolma-cc-head_0010.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18455249471 ${DATASET_DIR}/train/en/dolma-cc-head_0011.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18359243399 ${DATASET_DIR}/train/en/dolma-cc-head_0012.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16268609444 ${DATASET_DIR}/train/en/dolma-cc-head_0013.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15209913539 ${DATASET_DIR}/train/en/dolma-cc-head_0014.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15601099503 ${DATASET_DIR}/train/en/dolma-cc-head_0015.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16354139164 ${DATASET_DIR}/train/en/dolma-cc-head_0016.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19563123039 ${DATASET_DIR}/train/en/dolma-cc-head_0017.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17794386584 ${DATASET_DIR}/train/en/dolma-cc-head_0018.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17974377563 ${DATASET_DIR}/train/en/dolma-cc-head_0019.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19152181306 ${DATASET_DIR}/train/en/dolma-cc-head_0020.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16841018460 ${DATASET_DIR}/train/en/dolma-cc-head_0021.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15622566364 ${DATASET_DIR}/train/en/dolma-cc-head_0022.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14998264524 ${DATASET_DIR}/train/en/dolma-cc-head_0023.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19994706100 ${DATASET_DIR}/train/en/dolma-cc-head_0024.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19266785326 ${DATASET_DIR}/train/en/dolma-cc-head_0025.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17797970694 ${DATASET_DIR}/train/en/dolma-cc-head_0026.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18662607705 ${DATASET_DIR}/train/en/dolma-cc-head_0027.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18428148263 ${DATASET_DIR}/train/en/dolma-cc-head_0028.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19152709797 ${DATASET_DIR}/train/en/dolma-cc-head_0029.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19567672702 ${DATASET_DIR}/train/en/dolma-cc-head_0030.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15453203385 ${DATASET_DIR}/train/en/dolma-cc-head_0031.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16946844380 ${DATASET_DIR}/train/en/dolma-cc-head_0032.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16719501611 ${DATASET_DIR}/train/en/dolma-cc-head_0033.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16348054343 ${DATASET_DIR}/train/en/dolma-cc-head_0034.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18292316049 ${DATASET_DIR}/train/en/dolma-cc-head_0035.jsonl_text_document"

# en dolma science paper
DATA_PATH="${DATA_PATH} 8089227423 ${DATASET_DIR}/train/en/dolma-pes2o_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 20185217235 ${DATASET_DIR}/train/en/dolma-pes2o_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18622836173 ${DATASET_DIR}/train/en/dolma-pes2o_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15956491971 ${DATASET_DIR}/train/en/dolma-pes2o_0003.jsonl_text_document"

# en dolma reddit
DATA_PATH="${DATA_PATH} 17412289508 ${DATASET_DIR}/train/en/dolma-reddit_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17315996345 ${DATASET_DIR}/train/en/dolma-reddit_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17095921975 ${DATASET_DIR}/train/en/dolma-reddit_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15808400388 ${DATASET_DIR}/train/en/dolma-reddit_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15425532535 ${DATASET_DIR}/train/en/dolma-reddit_0004.jsonl_text_document"

# en dolma wiki
DATA_PATH="${DATA_PATH} 3896965449 ${DATASET_DIR}/train/en/dolma-wiki_0000.jsonl_text_document"

# en wiki
DATA_PATH="${DATA_PATH} 4744259830 ${DATASET_DIR}/train/en/wiki_0000.jsonl_text_document"

# zh wiki
DATA_PATH="${DATA_PATH} 840277331 ${DATASET_DIR}/train/zh/wiki_0000.jsonl_text_document"

# ko wiki
DATA_PATH="${DATA_PATH} 316296219 ${DATASET_DIR}/train/ko/wiki_0000.jsonl_text_document"

# Model arguments
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${SEQ_LENGTH}
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --num-attention-heads ${NUM_HEADS}
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --num-query-groups 16
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 10000
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --data-cache-path $CACHE_DIR
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --lr ${LR}
    --train-iters ${TRAIN_STEPS}
    --lr-decay-iters ${TRAIN_STEPS}
    --lr-decay-style cosine
    --min-lr ${MIN_LR}
    --weight-decay ${WEIGHT_DECAY}
    --lr-warmup-iters ${LR_WARMUP_STEPS}
    --clip-grad ${GRAD_CLIP}
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --bf16
    --use-flash-attn
    --transformer-impl "transformer_engine"
    --attention-softmax-in-fp32
    --distributed-backend nccl
)

# Model parameters
MODEL_PARALLEL_ARGS=(
   --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE}
   --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE}
   --expert-model-parallel-size ${EXPERT_PARALLEL_SIZE}
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
   --use-distributed-optimizer
   --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --moe-per-layer-logging
    --save-interval 500
    --eval-interval 500
    --eval-iters 10
    --save ${CHECKPOINT_PATH}
    --load ${CHECKPOINT_PATH}
    --use-mpi
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name ${WANDB_NAME}
)


export NVTE_FUSED_ATTN=0
mpirun \
    -np $NUM_GPUS \
    --npernode $NUM_GPUS_PER_NODE \
    -bind-to none \
    -map-by slot \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x NUM_NODES=$NUM_NODES \
    -x NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
    python pretrain_gpt.py \
        "${MODEL_ARGS[@]}" \
        "${MOE_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${LOGGING_ARGS[@]}"
