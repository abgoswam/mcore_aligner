#!/bin/bash
if [ -z "$JOB_PATH" ]; then
  MCORE_ALIGNER_PATH=/home/aiscuser/mcore_aligner
else
  # this is amulet. so JOB_PATH will be set, and code has been uploaded. 
  MCORE_ALIGNER_PATH=.
fi

# simple_gpt_batch_inference.py is located here.
cd ${MCORE_ALIGNER_PATH}/AMA-Megatron-LM-10152024/examples/inference/gpt
ls -lh

# setting PYTHONPATH so that simple_gpt_batch_inference.py get to see megatron.
MEGATRON_PATH=${MCORE_ALIGNER_PATH}/AMA-Megatron-LM-10152024
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

# =============================================
set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1

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
    --num-tokens-to-generate 64
    --max-batch-size 4
)

# 1 GPU
# python simple_gpt_batch_inference.py \
#     ${TOKENIZER_ARGS[@]} \
#     ${MODEL_ARGS[@]} \
#     ${INFERENCE_SPECIFIC_ARGS[@]} \
#     --prompts  "Swiss-born novelist and poet (1887–1961)" "Swiss-born novelist and poet (1887–1961)  was a member of the Royal" "Swiss-born novelist and poet (1887–1961)  was a member of the Royal Navy"

# Multi-GPU.
torchrun --nproc-per-node=4 simple_gpt_batch_inference.py \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]} \
    --prompts  "Swiss-born novelist and poet (1887–1961)" "Swiss-born novelist and poet (1887–1961)  was a member of the Royal" "Swiss-born novelist and poet (1887–1961)  was a member of the Royal Navy"