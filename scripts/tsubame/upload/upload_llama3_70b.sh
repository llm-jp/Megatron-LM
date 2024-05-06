#!/bin/sh
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=24:00:00
#$ -o outputs/upload/llama-3-70b/$JOB_ID
#$ -e outputs/upload/llama-3-70b/$JOB_ID
#$ -p -5

set -e

source .env/bin/activate

start=2000
end=2000
increment=500

base_dirs=(
  "/gs/bs/tgh-NII-LLM/checkpoints/megatron-to-hf/Llama-3-70b-hf/LR1.0e-5-MINLR1.0E-6-WD0.05"
  "/gs/bs/tgh-NII-LLM/checkpoints/megatron-to-hf/Llama-3-70b-hf/LR2.5e-5-MINLR2.5E-6-WD0.05"
  "/gs/bs/tgh-NII-LLM/checkpoints/megatron-to-hf/Llama-3-70b-hf/LR2.5e-5-MINLR2.5E-6-WD0.1"
)

for base_dir in "${base_dirs[@]}"; do
  for ((i = start; i <= end; i += increment)); do
    upload_dir=$base_dir/iter_$(printf "%07d" $i)

    python scripts/abci/upload/upload.py \
      --ckpt-path $upload_dir \
      --repo-name tokyotech-llm/Llama3-70b-$(basename $base_dir)-iter$(printf "%07d" $i)
  done
done
