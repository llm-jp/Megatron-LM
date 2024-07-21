#!/bin/bash
#SBATCH --job-name=upload
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --output=outputs/auto-ckpt-upload/%x-%j.out
#SBATCH --error=outputs/auto-ckpt-upload/%x-%j.out

BUCKET_NAME="llama2-172b-checkpoint-exp2"
CHECKPOINT_DIR="/lustre/checkpoints/llama-2-172b-exp2/tp4-pp16-cp1"

cd $CHECKPOINT_DIR

while : ;do
	LATEST_ITER=$(printf  %07d $(cat latest_checkpointed_iteration.txt))
	gsutil -m rsync -r iter_${LATEST_ITER} gs://${BUCKET_NAME}/iter_${LATEST_ITER}
	sleep 10m
done
