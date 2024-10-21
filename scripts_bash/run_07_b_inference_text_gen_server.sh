#!/bin/bash
if [ -z "$JOB_PATH" ]; then
  MCORE_ALIGNER_PATH=/home/aiscuser/mcore_aligner
else
  # this is amulet. so JOB_PATH will be set, and code has been uploaded. 
  MCORE_ALIGNER_PATH=.
fi

# simple_gpt_batch_inference.py is located here.
cd ${MCORE_ALIGNER_PATH}/AMA-Megatron-LM-10152024/
ls -lh

# setting PYTHONPATH so that simple_gpt_batch_inference.py get to see megatron.
MEGATRON_PATH=${MCORE_ALIGNER_PATH}/AMA-Megatron-LM-10152024
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

# =============================================

#!/bin/bash
set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

# This example will start serving the 345M model.
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"


export CUDA_DEVICE_MAX_CONNECTIONS=1

pip install flask-restful

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model mistralai/Mistral-7B-v0.1
)

MODEL_ARGS=(
    --use-checkpoint-args
    --use-mcore-models
    --load "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced/output_mcore_mistral_sft5/checkpoint/dsw-finetune-megatron-gpt3-7B-lr-1e-5-bs-1-seqlen-128-pr-bf16-tp-1-pp-1-ac-sel-do-true-sp-false-ti-100-wi-10"
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]} \
    --micro-batch-size 1 \
    --seed 42