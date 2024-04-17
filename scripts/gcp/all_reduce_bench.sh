#!/bin/bash
#SBATCH --job-name=all-reduce
#SBATCH --time=0:30:00
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 70
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/all-reduce-pef/%x-%j.out
#SBATCH --error=outputs/all-reduce-pef/%x-%j.out

# module load
module load cuda/12.1
module load cudnn/8.9.7
module load hpcx/2.17.1

# python virtualenv
source .env/bin/activate

# Usage: sbatch run-nccl-tests.sh
echo Running NCCL Tests on ${SLURM_JOB_NUM_NODES} nodes.

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
  export NCCL_MAX_NCHANNELS=12
  export NCCL_MIN_NCHANNELS=12
  export NCCL_DYNAMIC_CHUNK_SIZE=524288
  export NCCL_P2P_NET_CHUNKSIZE=524288
  export NCCL_P2P_PCI_CHUNKSIZE=524288
  export NCCL_P2P_NVL_CHUNKSIZE=1048576
  export NCCL_BUFFSIZE=4194304
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export NCCL_NET_GDR_LEVEL=PIX
  export NCCL_P2P_PXN_LEVEL=0
  export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=${UDS_PATH}
  export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000
  export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
  export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=1000

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

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -bind-to none -map-by slot \
  -x PATH \
  python scripts/gcp/all_reduce_bench.py
