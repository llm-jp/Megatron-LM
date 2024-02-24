#!/bin/bash
#SBATCH --job-name=all-reduce
#SBATCH --time=0:30:00
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out

# module load
module load cuda/12.1
module load cudnn/8.9.7

# python virtualenv
source .env/bin/activate

# distributed settings
# export MASTER_ADDR=$(/usr/sbin/ip addr show enp0s12 | grep "inet " | awk '{print $2}' | cut -d/ -f1)
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="H100"


NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -bind-to none -map-by slot \
  -x PATH \
  python scripts/gcp/all_reduce_bench.py
