#!/bin/bash
#SBATCH --job-name=llama-2-13b
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/llama-2-13b/%x-%j.out
#SBATCH --error=outputs/llama-2-13b/%x-%j.out

set -e

# module load
module load cuda/12.1
module load cudnn/8.9.7
module load hpcx/2.17.1

# open file limit
ulimit -n 65536 1048576

# python virtualenv
source venv/bin/activate

# Important TCPX environment variables
UDS_PATH="/run/tcpx-${SLURM_JOB_ID}"

# Only use TCPX for multi-node jobs.
[[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]] && export USE_TCPX=yes || export USE_TCPX=no

# Only use TCPX for multi-node jobs.
if [[ ${USE_TCPX} = "yes" ]]; then
  # Set up NCCL Environment variables
  export NCCL_NET=GPUDirectTCPX_v7
  # These network interfaces use Ubuntu's consistent naming scheme. See
  # https://manpages.ubuntu.com/manpages/focal/man7/systemd.net-naming-scheme.7.html
  export NCCL_SOCKET_IFNAME=enp0s12
  export NCCL_GPUDIRECTTCPX_CTRL_DEV=enp0s12
  export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=enp6s0,enp12s0,enp134s0,enp140s0
  export NCCL_CROSS_NIC=0
  export NCCL_ALGO=Ring
  export NCCL_PROTO=Simple
  export NCCL_NSOCKS_PERTHREAD=4
  export NCCL_SOCKET_NTHREADS=1
  export NCCL_DYNAMIC_CHUNK_SIZE=524288
  export NCCL_P2P_NET_CHUNKSIZE=524288
  export NCCL_P2P_PCI_CHUNKSIZE=524288
  export NCCL_P2P_NVL_CHUNKSIZE=1048576
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export NCCL_NET_GDR_LEVEL=PIX
  export NCCL_P2P_PXN_LEVEL=0
  export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=${UDS_PATH}
  export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=500000
  export NCCL_GPUDIRECTTCPX_TX_BINDINGS="enp6s0:8-21,112-125;enp12s0:8-21,112-125;enp134s0:60-73,164-177;enp140s0:60-73,164-177"
  export NCCL_GPUDIRECTTCPX_RX_BINDINGS="enp6s0:22-35,126-139;enp12s0:22-35,126-139;enp134s0:74-87,178-191;enp140s0:74-87,178-191"

  export LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH}
else
  unset NCCL_NET
fi

# The following two can be useful for debugging
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING

# distributed settings
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="H100"


NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))


# model config
# llama-2-13b: https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
HIDDEN_SIZE=5120
FFN_HIDDEN_SIZE=13824 # intermediate size (HuggingFace)
NUM_LAYERS=40
NUM_HEADS=40
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=2  # fixed
PIPELINE_PARALLEL_SIZE=2 # num layers 40: Llama-2 13B
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=61000
LR_DECAY_ITERS=61000

LR=3e-4
MIN_LR=3E-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/home/ext_kazuki_fujii_rio_gsic_titech/llm-jp-tokenizer/models/ver2.2/code20K_en40K_ja60K.ver2.2.model
CHECKPOINT_SAVE_DIR=/home/ext_taishi_nakamura_rio_gsic_tit/checkpoints/Llama-2-13b/exp3_v2_datasets_b2/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/home/ext_kazuki_fujii_rio_gsic_titech/datasets

TRAIN_DATA_PATH=""

TRAIN_DATA_PATH=""

TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1442858730 ${DATASET_DIR}/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/ja_wiki_Llama2Tokenizer/ja_wiki_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4697570389 ${DATASET_DIR}/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_wiki_Llama2Tokenizer/en_wiki_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8670888578 ${DATASET_DIR}/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/code_stack_Llama2Tokenizer/code_stack_merge_1.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16802680411 ${DATASET_DIR}/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_1.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16814033199 ${DATASET_DIR}/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_2.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16941076697 ${DATASET_DIR}/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_3.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16933151184 ${DATASET_DIR}/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_4.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16863221834 ${DATASET_DIR}/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_5.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16928046076 ${DATASET_DIR}/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_6.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8477979936 ${DATASET_DIR}/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile_Llama2Tokenizer/en_pile_merge_7.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 485412592.3 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 671720325.2 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1520327461 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2210405544 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1950073735 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2447399807 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1607587452 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2614097085 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1936833706 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2072378019 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1085097659 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2017-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1520398698 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2821380916 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1386476874 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1365024447 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1079992869 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3154362631 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3508566438 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2526846091 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2568027376 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2795873847 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2691462512 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3000093885 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2018-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2623504180 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2809876295 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2200639273 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2094244433 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-18.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2099166494 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1974100479 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1881839207 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2135269364 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-35.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1798071010 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1996550453 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1719580748 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1603557847 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2019-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2099920626 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2067348539 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-10.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1993241361 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-16.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2029791266 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-24.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2322944978 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-29.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1965010132 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2479626171 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-40.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1800491331 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-45.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1675306449 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2020-50.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2155870225 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1835666333 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-10.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2059578946 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1805208879 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-21.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1562020823 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-25.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1776641448 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-31.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1867977822 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2291113661 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2296646892 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2021-49.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2849405378 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2730361649 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-21.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2417978889 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-27.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2101837374 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-33.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1955769700 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-40.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2110014918 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2022-49.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2197497215 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2023-06.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1827420392 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2023-14.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1800224491 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2023-23.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2016585896 ${DATASET_DIR}/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1_Llama2Tokenizer/CC-MAIN-2013-2016.jsonl_text_document"

# job name
JOB_NAME="llama-2-13b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss-overlap-param-gather-grad-reduce"

# --norm-epsilon 1e-5 : conifg.json (RMS norm)

CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -bind-to none -map-by slot \
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
  --adam-eps 1e-8 \
  --log-interval 1 \
  --save-interval 500 \
  --eval-interval 500 \
  --eval-iters 10 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --position-embedding-type rope \
  --no-position-embedding \
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
  --use-mpi \
  --use-z-loss \
  --log-throughput \
  --distributed-timeout-minutes 15 \
  --wandb-name ${JOB_NAME} \
  --wandb-project "Llama-2-13B-debug-v2-datasets" \
  --wandb-entity "nii-geniac" \
