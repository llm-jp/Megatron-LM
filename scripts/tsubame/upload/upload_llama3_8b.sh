#!/bin/sh
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=24:00:00
#$ -o outputs/upload/llama-3-8b/$JOB_ID
#$ -e outputs/upload/llama-3-8b/$JOB_ID
#$ -p -5

set -e

source .env/bin/activate

start=12500
end=12500
increment=2500

base_dirs=(
  "/gs/bs/tgh-NII-LLM/checkpoints/megatron-to-hf/Llama-3-8b-hf/exp2/LR1.0E-5-MINLR1.0E-6-WD0.1-WARMUP1000"
)

for base_dir in "${base_dirs[@]}"; do
  for ((i = start; i <= end; i += increment)); do
    upload_dir=$base_dir/iter_$(printf "%07d" $i)

    python scripts/abci/upload/upload.py \
      --ckpt-path $upload_dir \
      --repo-name tokyotech-llm/Llama3-8b-exp2-$(basename $base_dir)-iter$(printf "%07d" $i)
  done
done
