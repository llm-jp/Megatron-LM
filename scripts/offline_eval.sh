# model config
# llama-2-13b: https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
HIDDEN_SIZE=5120
FFN_HIDDEN_SIZE=13824 # intermediate size (HuggingFace)
NUM_LAYERS=40
NUM_HEADS=40
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=500679
LR_DECAY_ITERS=452995

LR=2.5e-4
MIN_LR=2.5E-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/model/kouta/model/megatron/llm-jp-13B-v3/hf/tokenizer.model

CHECKPOINT_ARGS="--load /model/kouta/model/megatron/llm-jp-13B-v3_tp2_pp1"

launch_config="--nproc_per_node=8"

echo "Launch config: ${launch_config}"

export CUDA_DEVICE_MAX_CONNECTIONS=1

offline_eval_options=" \
    --offline-batch-size 10 \
    --dataset_dir /model/kouta/2024/MGMNInfe/llm-jp-eval/offline/prompts \
    --offline-output-dir /model/kouta/2024/MGMNInfe/llm-jp-eval/offline/_megatron-GENIAC-Llama2-13b-iter0146000"

torchrun ${launch_config} examples/inference/offline_eval.py \
    --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
    --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
    --use-distributed-optimizer \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    ${CHECKPOINT_ARGS} \
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
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 500 \
    --eval-iters 10 \
    --bf16 \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
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
    --log-throughput \
    --no-async-tensor-model-parallel-allreduce \
    --max-tokens-to-oom 24000 \
    ${offline_eval_options}