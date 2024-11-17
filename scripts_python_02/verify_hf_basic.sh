#!/bin/bash

set -x # Enable debugging
set -e # Exit on errors

ENV_VARS=(
    NCCL_DEBUG=WARN
    CUDA_DEVICE_MAX_CONNECTIONS=1
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    PYTHONPATH=$PYTHONPATH:/home/aiscuser/mcore_aligner/AMA-Megatron-LM-10152024
)

DISTRIBUTED_ARGS=()

SEQ_LEN=8192
GPT_MODEL_ARGS=(
    # --transformer-impl local # local will not work for RMSNorm
    --transformer-impl transformer_engine
    --normalization RMSNorm
    --num-layers 32 
    --hidden-size 4096
    --ffn-hidden-size 14336 
    --num-attention-heads 32 
    --seq-length $SEQ_LEN 
    --max-position-embeddings $SEQ_LEN 
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 10000
    --swiglu
    --untie-embeddings-and-output-weights
    --no-position-embedding
    --disable-bias-linear
    --group-query-attention
	--num-query-groups 8
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1 
    --bf16
    --use-flash-attn
)

MODEL_PARALLEL_ARGS=()

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model mistralai/Mistral-7B-Instruct-v0.1 
)

EVAL_AND_LOGGING_ARGS=()

# Construct the command
cmd="${ENV_VARS[@]} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    /home/aiscuser/mcore_aligner/scripts_python_02/verify_hf_basic.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}"

# Debugging: Print the command before executing
echo "$cmd"

# Execute the command
eval "$cmd"
