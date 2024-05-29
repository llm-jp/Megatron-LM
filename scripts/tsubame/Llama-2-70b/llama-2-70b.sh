#!/bin/sh
#$ -cwd
#$ -l node_f=40
#$ -l h_rt=96:00:00
#$ -o outputs/Llama-2-70b/$JOB_ID
#$ -e outputs/Llama-2-70b/$JOB_ID
#$ -p -5

set -e

# module load
module use /gs/bs/tga-bayes-crest/fujii/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.1/2.18.3
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# python virtualenv
source venv/bin/activate

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
# https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json
HIDDEN_SIZE=8192
FFN_HIDDEN_SIZE=28672 # intermediate size (HuggingFace)
NUM_LAYERS=80
NUM_HEADS=64
NUM_KEY_VALUE_HEADS=8
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=4   # fixed
PIPELINE_PARALLEL_SIZE=4  # num layers 80
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1000
TRAIN_STEPS=512696 # 2.1 T Tokens
LR_DECAY_ITERS=512696 # 2.1 T Tokens

LR=1e-4
MIN_LR=1e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/gs/fs/tga-bayes-crest/taishi/workspace/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
CHECKPOINT_SAVE_DIR=/gs/bs/tgh-NII-LLM/checkpoints/Llama-2-70b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0

TRAIN_DATA_PATH=""

# code stack
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14486363187 ${DATASET_DIR}/train/code/stack_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12799385151 ${DATASET_DIR}/train/code/stack_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17282923545 ${DATASET_DIR}/train/code/stack_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8861329235 ${DATASET_DIR}/train/code/stack_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6713413649 ${DATASET_DIR}/train/code/stack_0004.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8976432285 ${DATASET_DIR}/train/code/stack_0005.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17961273649 ${DATASET_DIR}/train/code/stack_0006.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12016948303 ${DATASET_DIR}/train/code/stack_0007.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14953094719 ${DATASET_DIR}/train/code/stack_0008.jsonl_text_document"

# ja cc 1
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 23783124862 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 36378129564 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 35477545812 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 35917231868 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 46203062776 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0004.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 40396278536 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0005.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 33444216206 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0006.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 32375495374 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0007.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 36068919622 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0008.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 26274952324 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0009.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 24024422756 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0010.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 34590145510 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0011.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 29567301906 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0012.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 26690562242 /gs/fs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0013.jsonl_text_document"

# ja cc 2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 35813749376 ${DATASET_DIR}/train/ja/cc-2_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 40034668924 ${DATASET_DIR}/train/ja/cc-2_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 31191828858 ${DATASET_DIR}/train/ja/cc-2_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 25086109508 ${DATASET_DIR}/train/ja/cc-2_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18979589830 ${DATASET_DIR}/train/ja/cc-2_0004.jsonl_text_document"

# ja cc 3
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 40987803038 ${DATASET_DIR}/train/ja/cc-3_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 41333549162 ${DATASET_DIR}/train/ja/cc-3_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 29810274406 ${DATASET_DIR}/train/ja/cc-3_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 22787733940 ${DATASET_DIR}/train/ja/cc-3_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15544493906 ${DATASET_DIR}/train/ja/cc-3_0004.jsonl_text_document"

# ja kaken
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1826105478 ${DATASET_DIR}/train/ja/kaken_0000.jsonl_text_document"

# ja warp html
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1329440698 ${DATASET_DIR}/train/ja/warp-html-01-06_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1397268214 ${DATASET_DIR}/train/ja/warp-html-07-12_0000.jsonl_text_document"

# ja warp pdf
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 33073405454 ${DATASET_DIR}/train2/ja/warp-pdf-e00_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 33058213404 ${DATASET_DIR}/train2/ja/warp-pdf-e00_0001.jsonl_text_document"

# ja warp pdf 0.2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18574732812 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15982866546 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15177942198 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18034242459 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18081506330 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0004.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18177981727 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0005.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16307187923 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0006.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17515189522 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0007.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17124960630 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0008.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18069428959 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0009.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11893446595 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0010.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17604601685 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0011.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13412364977 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0012.jsonl_text_document"

# ja wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2563804308 ${DATASET_DIR}/train/ja/wiki_0000.jsonl_text_document"

# en dolma books
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5494262694 ${DATASET_DIR}/train/en/dolma-books_0000.jsonl_text_document"

# en dolma c4
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17052861266 ${DATASET_DIR}/train/en/dolma-c4_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17051260422 ${DATASET_DIR}/train/en/dolma-c4_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17056648148 ${DATASET_DIR}/train/en/dolma-c4_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17057773049 ${DATASET_DIR}/train/en/dolma-c4_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17047888113 ${DATASET_DIR}/train/en/dolma-c4_0004.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17046511755 ${DATASET_DIR}/train/en/dolma-c4_0005.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17058086815 ${DATASET_DIR}/train/en/dolma-c4_0006.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17049490900 ${DATASET_DIR}/train/en/dolma-c4_0007.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17051009552 ${DATASET_DIR}/train/en/dolma-c4_0008.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14932405246 ${DATASET_DIR}/train/en/dolma-c4_0009.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13142696712 ${DATASET_DIR}/train/en/dolma-c4_0010.jsonl_text_document"

# en dolma cc
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15473522696 ${DATASET_DIR}/train/en/dolma-cc-head_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15767913273 ${DATASET_DIR}/train/en/dolma-cc-head_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16664785078 ${DATASET_DIR}/train/en/dolma-cc-head_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16860035920 ${DATASET_DIR}/train/en/dolma-cc-head_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17197613512 ${DATASET_DIR}/train/en/dolma-cc-head_0004.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16363353173 ${DATASET_DIR}/train/en/dolma-cc-head_0005.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15303692924 ${DATASET_DIR}/train/en/dolma-cc-head_0006.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15766283829 ${DATASET_DIR}/train/en/dolma-cc-head_0007.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13483997219 ${DATASET_DIR}/train/en/dolma-cc-head_0008.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12561851173 ${DATASET_DIR}/train/en/dolma-cc-head_0009.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14206017429 ${DATASET_DIR}/train/en/dolma-cc-head_0010.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18455249471 ${DATASET_DIR}/train/en/dolma-cc-head_0011.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18359243399 ${DATASET_DIR}/train/en/dolma-cc-head_0012.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16268609444 ${DATASET_DIR}/train/en/dolma-cc-head_0013.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15209913539 ${DATASET_DIR}/train/en/dolma-cc-head_0014.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15601099503 ${DATASET_DIR}/train/en/dolma-cc-head_0015.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16354139164 ${DATASET_DIR}/train/en/dolma-cc-head_0016.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19563123039 ${DATASET_DIR}/train/en/dolma-cc-head_0017.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17794386584 ${DATASET_DIR}/train/en/dolma-cc-head_0018.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17974377563 ${DATASET_DIR}/train/en/dolma-cc-head_0019.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19152181306 ${DATASET_DIR}/train/en/dolma-cc-head_0020.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16841018460 ${DATASET_DIR}/train/en/dolma-cc-head_0021.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15622566364 ${DATASET_DIR}/train/en/dolma-cc-head_0022.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14998264524 ${DATASET_DIR}/train/en/dolma-cc-head_0023.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19994706100 ${DATASET_DIR}/train/en/dolma-cc-head_0024.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19266785326 ${DATASET_DIR}/train/en/dolma-cc-head_0025.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17797970694 ${DATASET_DIR}/train/en/dolma-cc-head_0026.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18662607705 ${DATASET_DIR}/train/en/dolma-cc-head_0027.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18428148263 ${DATASET_DIR}/train/en/dolma-cc-head_0028.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19152709797 ${DATASET_DIR}/train/en/dolma-cc-head_0029.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19567672702 ${DATASET_DIR}/train/en/dolma-cc-head_0030.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15453203385 ${DATASET_DIR}/train/en/dolma-cc-head_0031.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16946844380 ${DATASET_DIR}/train/en/dolma-cc-head_0032.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16719501611 ${DATASET_DIR}/train/en/dolma-cc-head_0033.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16348054343 ${DATASET_DIR}/train/en/dolma-cc-head_0034.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18292316049 ${DATASET_DIR}/train/en/dolma-cc-head_0035.jsonl_text_document"

# en dolma science paper
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8089227423 ${DATASET_DIR}/train/en/dolma-pes2o_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 20185217235 ${DATASET_DIR}/train/en/dolma-pes2o_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18622836173 ${DATASET_DIR}/train/en/dolma-pes2o_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15956491971 ${DATASET_DIR}/train/en/dolma-pes2o_0003.jsonl_text_document"

# en dolma reddit
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17412289508 ${DATASET_DIR}/train/en/dolma-reddit_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17315996345 ${DATASET_DIR}/train/en/dolma-reddit_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17095921975 ${DATASET_DIR}/train/en/dolma-reddit_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15808400388 ${DATASET_DIR}/train/en/dolma-reddit_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15425532535 ${DATASET_DIR}/train/en/dolma-reddit_0004.jsonl_text_document"

# en dolma wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3896965449 ${DATASET_DIR}/train/en/dolma-wiki_0000.jsonl_text_document"

# en wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4744259830 ${DATASET_DIR}/train/en/wiki_0000.jsonl_text_document"

# zh wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 840277331 ${DATASET_DIR}/train/zh/wiki_0000.jsonl_text_document"

# ko wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 316296219 ${DATASET_DIR}/train/ko/wiki_0000.jsonl_text_document"

VALID_DATA_PATH=""

VALID_DATA_PATH="${VALID_DATA_PATH} 73520 ${DATASET_DIR}/validation/code/stack_0000.jsonl_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 946215 ${DATASET_DIR}/validation/en/wiki_0000.jsonl_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 886814 ${DATASET_DIR}/validation/ja/wiki_0000.jsonl_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 5203 ${DATASET_DIR}/validation/ko/wiki_0000.jsonl_text_document"
VALID_DATA_PATH="${VALID_DATA_PATH} 6145 ${DATASET_DIR}/validation/zh/wiki_0000.jsonl_text_document"

# job name
JOB_NAME="TSUBAME4-Llama-2-70b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

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
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
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
  --adam-eps 1e-5 \
  --log-interval 1 \
  --save-interval 100 \
  --eval-interval 500 \
  --eval-iters 10 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
  --position-embedding-type rope \
  --rope-theta 10000.0 \
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
  --fp8-format hybrid \
  --use-mpi \
  --use-z-loss \
  --distributed-timeout-minutes 20 \ 
  --log-throughput \
  --wandb-name ${JOB_NAME} \
  --wandb-project "TSUBAME4-Llama-2-70b" \
  --wandb-entity "llm-jp"
