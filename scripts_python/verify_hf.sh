NUM_NODES=1
GPUS_PER_NODE=1

# config for multinode
MASTER_ADDR=node-0
MASTER_PORT=$((RANDOM/1000+6010)) # use random port 
RDZV_ID=$((RANDOM+1000)) # random id
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))


# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt

# TOKENIZER=CL100kBaseBPETokenizer
# TOKENIZER=GPT4oTokenizer

CUR_DIR=`dirname $0`
DATA_CONFIG_PATH=./projects/phi3_silica/data_phi3_phase_1_noacademic_ablation.json


TP_SIZE=1
PP_SIZE=1
DP_SIZE=$((WORLD_SIZE / (TP_SIZE * PP_SIZE)))
TOKENS_PER_GLOBAL_BATCH=$((1024 * 1024 * 4)) # 4M tokens as in phi-3-min
SEQ_LEN=4096 # was used in original phi-3-min
TARGET_GLOBAL_BATCH_SIZE=$((TOKENS_PER_GLOBAL_BATCH / SEQ_LEN))
MICRO_BATCH_SIZE=2


########################
### training horizon ###
########################
B=1000000000 # 1B
TOKENS_IN_BILL=200  # 200B for ablation study
TOKENS=$(( TOKENS_IN_BILL * B ))  # 300B tokens
TRAIN_ITERS=$(( TOKENS / (TARGET_GLOBAL_BATCH_SIZE * SEQ_LEN) ))

if [[ $TARGET_GLOBAL_BATCH_SIZE -le $((MICRO_BATCH_SIZE * DP_SIZE))  ]]; then
    echo "TARGET_GLOBAL_BATCH_SIZE ($TARGET_GLOBAL_BATCH_SIZE) is smaller than micro_batch ($MICRO_BATCH_SIZE) * dp_size ($DP_SIZE)"
    exit
fi


##############################
### NAME/CHECKPOINT/TB/LOG ###
##############################


MODEL_ROOT=/mnt/std-cache/users/xihlin/tmp/megatron_lm/
LOG_ROOT=/data/users/xihlin/tmp/megatron_lm_tb
PROJECT_NAME=2024-10-24-phi3_silica-tp1pp1-200b-gbs8388608-mbs2-adam_eps_1e-7-untied_emb-rope10k
CHECKPOINT_PATH=$MODEL_ROOT/${PROJECT_NAME} #<Specify path>
TENSORBOARD_LOG_PATH=$LOG_ROOT/${PROJECT_NAME} #<Specify path>


DISTRIBUTED_ARGS=(
    --rdzv_id=${RDZV_ID}
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
    --rdzv_backend=c10d
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$NUM_NODES 
)

GPT_MODEL_ARGS=(
    # --transformer-impl local # local will not work for RMSNorm
    --transformer-impl transformer_engine
    --normalization RMSNorm
    --num-layers 32 
    --hidden-size 3072 
    --num-attention-heads 32 
    --seq-length $SEQ_LEN 
    --max-position-embeddings $SEQ_LEN 
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 10000
    --swiglu
    --untie-embeddings-and-output-weights
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $TARGET_GLOBAL_BATCH_SIZE 
    --bf16
    --use-flash-attn
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size ${TP_SIZE}
	--pipeline-model-parallel-size ${PP_SIZE}
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model microsoft/Phi-3-mini-4k-instruct 
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --log-throughput
    --log-progress
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --load $CHECKPOINT_PATH 
    --tensorboard-dir $TENSORBOARD_LOG_PATH.eval 
)

ENV_VARS=(
    NCCL_DEBUG=WARN
    CUDA_DEVICE_MAX_CONNECTIONS=1
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    PYTHONPATH=$PYTHONPATH:.
)

cmd="${ENV_VARS[@]} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    projects/phi3_silica/verify_hf.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}"


eval $cmd