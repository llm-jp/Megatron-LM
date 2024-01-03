#!/bin/bash

cd /mnt/taishi-work-space/nccl-tests

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=12345

HOSTFILE_NAME=/mnt/taishi-work-space/Megatron-LM/hostfile/2node

mpirun -np 2 \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x LD_LIBRARY_PATH=/mnt/taishi-work-space/nccl/build/lib/:$LD_LIBRARY_PATH \
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8