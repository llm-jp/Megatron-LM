#!/bin/bash
#SBATCH --job-name=0059_v3-8x1.8b-exp1
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
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

TOKENIZER_MODEL="llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model"
MEGATRON_PATH="/home/taishi/Megatron-LM"
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH:-}

TARGET_TP_SIZE=$1
TARGET_EP_SIZE=$2
TARGET_PP_SIZE=$3

HF_FORMAT_DIR=/home/shared/experiments/0061_v3-8x1.8b-exp2/Mixtral-llm-jp-v3-8x1.8B-initial-checkpoint_lam-las-ind
MEGATRON_FORMAT_DIR=mcore-Mixtral-llm-jp-v3-8x1.8B-initial-checkpoint_lam-las-ind-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}
mkdir -p $MEGATRON_FORMAT_DIR

python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader mixtral_hf \
    --saver mcore \
    --target-tensor-parallel-size ${TARGET_TP_SIZE} \
    --target-pipeline-parallel-size ${TARGET_PP_SIZE} \
    --target-expert-parallel-size ${TARGET_EP_SIZE} \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --true-vocab-size 99574 \
    --tokenizer-model ${TOKENIZER_MODEL}
