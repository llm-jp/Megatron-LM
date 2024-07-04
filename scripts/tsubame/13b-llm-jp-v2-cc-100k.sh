#!/bin/sh
#$ -cwd
#$ -l node_f=32
#$ -l h_rt=10:00:00:00
#$ -o outputs/Llama-2-13b/$JOB_ID
#$ -e outputs/Llama-2-13b/$JOB_ID
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
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
TENSOR_PARALLEL_SIZE=4   # fixed
PIPELINE_PARALLEL_SIZE=1 # num layers 40: Llama-2 13B
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=61000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
# 今回は約270B Tokens #実際の数値は250B Tokens

LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/gs/bs/tga-NII-LLM/src/llm-jp-tokenizer/models/ver2.2/code20K_en40K_ja60K.ver2.2.model
CHECKPOINT_SAVE_DIR=/gs/bs/tgh-NII-LLM/checkpoints/Llama-2-13b/CC_v2_code20K_en40K_ja60K_ver2.2/exp-A/

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code20K_en40K_ja60K.ver2.2

TRAIN_DATA_PATH=""

# code stack
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14705093006 ${DATASET_DIR}/train/code/stack_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12897462302 ${DATASET_DIR}/train/code/stack_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17375285594 ${DATASET_DIR}/train/code/stack_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8969185145 ${DATASET_DIR}/train/code/stack_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6827765273 ${DATASET_DIR}/train/code/stack_0004.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9363975453 ${DATASET_DIR}/train/code/stack_0005.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19026599589 ${DATASET_DIR}/train/code/stack_0006.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12282415138 ${DATASET_DIR}/train/code/stack_0007.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15263156901 ${DATASET_DIR}/train/code/stack_0008.jsonl_text_document"

# japanese cc-1
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 29991427846 ${DATASET_DIR}/train/ja/cc-1_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 45927020954 ${DATASET_DIR}/train/ja/cc-1_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 44797836792 ${DATASET_DIR}/train/ja/cc-1_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 45413856466 ${DATASET_DIR}/train/ja/cc-1_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 58539509136 ${DATASET_DIR}/train/ja/cc-1_0004.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 51170457202 ${DATASET_DIR}/train/ja/cc-1_0005.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 42358271956 ${DATASET_DIR}/train/ja/cc-1_0006.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 41030625798 ${DATASET_DIR}/train/ja/cc-1_0007.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 45756797772 ${DATASET_DIR}/train/ja/cc-1_0008.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 33383962016 ${DATASET_DIR}/train/ja/cc-1_0009.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 30450071474 ${DATASET_DIR}/train/ja/cc-1_0010.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 44262815366 ${DATASET_DIR}/train/ja/cc-1_0011.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 37406859854 ${DATASET_DIR}/train/ja/cc-1_0012.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 33603488934 ${DATASET_DIR}/train/ja/cc-1_0013.jsonl_text_document"

# japanese cc-2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 45098102062 ${DATASET_DIR}/train/ja/cc-2_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 50587129984 ${DATASET_DIR}/train/ja/cc-2_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 39530330518 ${DATASET_DIR}/train/ja/cc-2_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 31829557578 ${DATASET_DIR}/train/ja/cc-2_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 24035635422 ${DATASET_DIR}/train/ja/cc-2_0004.jsonl_text_document"

# japanese cc-3
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 51613167946 ${DATASET_DIR}/train/ja/cc-3_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 52227162096 ${DATASET_DIR}/train/ja/cc-3_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 37795893374 ${DATASET_DIR}/train/ja/cc-3_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 28950667002 ${DATASET_DIR}/train/ja/cc-3_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19725854052 ${DATASET_DIR}/train/ja/cc-3_0004.jsonl_text_document"

# japanese kaken
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2212144648 ${DATASET_DIR}/train/ja/kaken_0000.jsonl_text_document"

# japanese warp html
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1672736752 ${DATASET_DIR}/train/ja/warp-html-01-06_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1758537968 ${DATASET_DIR}/train/ja/warp-html-07-12_0000.jsonl_text_document"

# japanese warp pdf 01
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 40505005868 ${DATASET_DIR}/train2/ja/warp-pdf-e00_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 40649405498 ${DATASET_DIR}/train2/ja/warp-pdf-e00_0001.jsonl_text_document"

# japanese warp pdf 02
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 22112928123 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19023096646 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18053192768 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 21421464164 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 21481501823 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0004.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 21630552732 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0005.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19362965666 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0006.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 20839654477 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0007.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 20380480552 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0008.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 21583438879 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0009.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14151453447 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0010.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 20978733957 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0011.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15813185027 ${DATASET_DIR}/train2/ja/warp-pdf-e02_0012.jsonl_text_document"

# japanese wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3090031262 ${DATASET_DIR}/train/ja/wiki_0000.jsonl_text_document"

# english dolma guten
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5430888964 ${DATASET_DIR}/train/en/dolma-books_0000.jsonl_text_document"

# english dolma c4
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17039419937 ${DATASET_DIR}/train/en/dolma-c4_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17038484023 ${DATASET_DIR}/train/en/dolma-c4_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17043451611 ${DATASET_DIR}/train/en/dolma-c4_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17044307492 ${DATASET_DIR}/train/en/dolma-c4_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17033940670 ${DATASET_DIR}/train/en/dolma-c4_0004.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17031979996 ${DATASET_DIR}/train/en/dolma-c4_0005.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17043742900 ${DATASET_DIR}/train/en/dolma-c4_0006.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17036705857 ${DATASET_DIR}/train/en/dolma-c4_0007.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17038383046 ${DATASET_DIR}/train/en/dolma-c4_0008.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14920518572 ${DATASET_DIR}/train/en/dolma-c4_0009.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13132407175 ${DATASET_DIR}/train/en/dolma-c4_0010.jsonl_text_document"

# english dolma cc
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15355077930 ${DATASET_DIR}/train/en/dolma-cc-head_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15648608817 ${DATASET_DIR}/train/en/dolma-cc-head_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16540772293 ${DATASET_DIR}/train/en/dolma-cc-head_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16742353707 ${DATASET_DIR}/train/en/dolma-cc-head_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17078189081 ${DATASET_DIR}/train/en/dolma-cc-head_0004.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16238037991 ${DATASET_DIR}/train/en/dolma-cc-head_0005.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15179008223 ${DATASET_DIR}/train/en/dolma-cc-head_0006.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15641393153 ${DATASET_DIR}/train/en/dolma-cc-head_0007.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13391302659 ${DATASET_DIR}/train/en/dolma-cc-head_0008.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12476498502 ${DATASET_DIR}/train/en/dolma-cc-head_0009.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14110114125 ${DATASET_DIR}/train/en/dolma-cc-head_0010.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18333186442 ${DATASET_DIR}/train/en/dolma-cc-head_0011.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18246653235 ${DATASET_DIR}/train/en/dolma-cc-head_0012.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16179594492 ${DATASET_DIR}/train/en/dolma-cc-head_0013.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15118454347 ${DATASET_DIR}/train/en/dolma-cc-head_0014.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15508677643 ${DATASET_DIR}/train/en/dolma-cc-head_0015.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16267446129 ${DATASET_DIR}/train/en/dolma-cc-head_0016.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19471098788 ${DATASET_DIR}/train/en/dolma-cc-head_0017.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17706006600 ${DATASET_DIR}/train/en/dolma-cc-head_0018.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17879931226 ${DATASET_DIR}/train/en/dolma-cc-head_0019.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19052000015 ${DATASET_DIR}/train/en/dolma-cc-head_0020.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16748132577 ${DATASET_DIR}/train/en/dolma-cc-head_0021.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15536077127 ${DATASET_DIR}/train/en/dolma-cc-head_0022.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14920129555 ${DATASET_DIR}/train/en/dolma-cc-head_0023.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19889169429 ${DATASET_DIR}/train/en/dolma-cc-head_0024.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19164538620 ${DATASET_DIR}/train/en/dolma-cc-head_0025.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17703821720 ${DATASET_DIR}/train/en/dolma-cc-head_0026.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18566514714 ${DATASET_DIR}/train/en/dolma-cc-head_0027.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18318859672 ${DATASET_DIR}/train/en/dolma-cc-head_0028.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19032096408 ${DATASET_DIR}/train/en/dolma-cc-head_0029.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19449972004 ${DATASET_DIR}/train/en/dolma-cc-head_0030.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15361989783 ${DATASET_DIR}/train/en/dolma-cc-head_0031.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16842288871 ${DATASET_DIR}/train/en/dolma-cc-head_0032.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16616376116 ${DATASET_DIR}/train/en/dolma-cc-head_0033.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16247498404 ${DATASET_DIR}/train/en/dolma-cc-head_0034.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18186659029 ${DATASET_DIR}/train/en/dolma-cc-head_0035.jsonl_text_document"

# english dolma sciennce paper
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7658304269 ${DATASET_DIR}/train/en/dolma-pes2o_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19938548235 ${DATASET_DIR}/train/en/dolma-pes2o_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18434984047 ${DATASET_DIR}/train/en/dolma-pes2o_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15796462317 ${DATASET_DIR}/train/en/dolma-pes2o_0003.jsonl_text_document"

# english dolma reddit
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17578949411 ${DATASET_DIR}/train/en/dolma-reddit_0000.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17482085708 ${DATASET_DIR}/train/en/dolma-reddit_0001.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17259122761 ${DATASET_DIR}/train/en/dolma-reddit_0002.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15958589832 ${DATASET_DIR}/train/en/dolma-reddit_0003.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15630836036 ${DATASET_DIR}/train/en/dolma-reddit_0004.jsonl_text_document"

# english dolma wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3909303063 ${DATASET_DIR}/train/en/dolma-wiki_0000.jsonl_text_document"

# english wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4799264925 ${DATASET_DIR}/train/en/wiki_0000.jsonl_text_document"

# zh wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1149288693 ${DATASET_DIR}/train/zh/wiki_0000.jsonl_text_document.bin wiki_0000.jsonl_text_document"

# ko wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1267380188 ${DATASET_DIR}/train/ko/wiki_0000.jsonl_text_document"


# job name
JOB_NAME="exp-A-13b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# --norm-epsilon 1e-5 : conifg.json (RMS norm)

# # checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
fi

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
  --data-path ${TRAIN_DATA_PATH} \
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
  --wandb-project "geniac-13b-debug" \
  --wandb-entity "llm-jp"

