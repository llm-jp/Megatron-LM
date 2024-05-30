#!/bin/bash
set -e
# python virtualenv
source .env/bin/activate

python scripts/abci/megatron_to_hf/check.py \
  --base-hf-model-path /model/checkpoints_1.3b_hf/CC_v2_code20K_en40K_ja60K_ver2.2/ \
  --converted-hf-model-path /model/checkpoints_1.3b_hf/megatron_to_hf/CC_v2_code20K_en40K_ja60K_ver2.2/ \
