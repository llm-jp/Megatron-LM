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
CHECKPOINT_SAVE_DIR=/data/checkpoints_7b/CC_v2_100k_ver3.0_z_loss_only_GENIAC

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATA_PATH=""
DATA_PATH="${DATA_PATH} 14486363187 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/code/stack_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 12799385151 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/code/stack_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17282923545 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/code/stack_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 8861329235 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/code/stack_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 6713413649 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/code/stack_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 8976432285 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/code/stack_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17961273649 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/code/stack_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 12016948303 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/code/stack_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14953094719 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/code/stack_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 23783124862 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 36378129564 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 35477545812 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 35917231868 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 46203062776 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 40396278536 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 33444216206 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 32375495374 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 36068919622 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 26274952324 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0009.jsonl_text_document"
DATA_PATH="${DATA_PATH} 24024422756 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0010.jsonl_text_document"
DATA_PATH="${DATA_PATH} 34590145510 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0011.jsonl_text_document"
DATA_PATH="${DATA_PATH} 29567301906 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0012.jsonl_text_document"
DATA_PATH="${DATA_PATH} 26690562242 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-1_0013.jsonl_text_document"
DATA_PATH="${DATA_PATH} 35813749376 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-2_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 40034668924 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-2_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 31191828858 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-2_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 25086109508 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-2_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18979589830 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-2_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 40987803038 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-3_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 41333549162 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-3_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 29810274406 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-3_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 22787733940 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-3_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15544493906 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/cc-3_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1826105478 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/kaken_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1329440698 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/warp-html-01-06_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 1397268214 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/warp-html-07-12_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 33073405454 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e00_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 33058213404 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e00_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18574732812 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15982866546 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15177942198 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18034242459 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18081506330 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18177981727 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16307187923 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17515189522 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17124960630 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18069428959 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0009.jsonl_text_document"
DATA_PATH="${DATA_PATH} 11893446595 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0010.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17604601685 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0011.jsonl_text_document"
DATA_PATH="${DATA_PATH} 13412364977 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train2/ja/warp-pdf-e02_0012.jsonl_text_document"
DATA_PATH="${DATA_PATH} 2563804308 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/wiki_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 5494262694 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-books_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17052861266 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17051260422 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17056648148 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17057773049 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17047888113 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17046511755 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17058086815 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17049490900 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17051009552 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14932405246 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0009.jsonl_text_document"
DATA_PATH="${DATA_PATH} 13142696712 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-c4_0010.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15473522696 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15767913273 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16664785078 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16860035920 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17197613512 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16363353173 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15303692924 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15766283829 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 13483997219 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 12561851173 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0009.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14206017429 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0010.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18455249471 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0011.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18359243399 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0012.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16268609444 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0013.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15209913539 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0014.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15601099503 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0015.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16354139164 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0016.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19563123039 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0017.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17794386584 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0018.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17974377563 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0019.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19152181306 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0020.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16841018460 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0021.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15622566364 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0022.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14998264524 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0023.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19994706100 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0024.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19266785326 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0025.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17797970694 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0026.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18662607705 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0027.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18428148263 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0028.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19152709797 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0029.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19567672702 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0030.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15453203385 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0031.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16946844380 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0032.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16719501611 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0033.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16348054343 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0034.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18292316049 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-cc-head_0035.jsonl_text_document"
DATA_PATH="${DATA_PATH} 8089227423 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-pes2o_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 20185217235 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-pes2o_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18622836173 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-pes2o_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15956491971 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-pes2o_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17412289508 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-reddit_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17315996345 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-reddit_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17095921975 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-reddit_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15808400388 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-reddit_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15425532535 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-reddit_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 3896965449 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/dolma-wiki_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 4744259830 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/en/wiki_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 840277331 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/zh/wiki_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 316296219 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ko/wiki_0000.jsonl_text_document"
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
  --use-checkpoint-args \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
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
  --swiglu \
  --use-flash-attn \
  --use-z-loss \
  --recompute-activations \
  --recompute-granularity "selective" \
  --use-mpi \
  --wandb-name ${JOB_NAME} \
  --wandb-project "megatron-lm-7B-2024-0303-ver3.0" \
  --wandb-entity "llm-jp"
