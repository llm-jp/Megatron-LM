export CUDA_DEVICE_MAX_CONNECTIONS=1

python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader mcore \
    --saver mcore \
    --load-dir /model/kouta/model/megatron/llm-jp-13B-v3 \
    --save-dir /model/kouta/model/megatron/llm-jp-13B-v3_tp2_pp1 \
    --megatron-path /model/kouta/2024/MGMNInfe/Megatron-LM \
    --loader-transformer-impl "transformer_engine" \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 1