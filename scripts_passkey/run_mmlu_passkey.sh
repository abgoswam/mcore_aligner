#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)

echo "==== Starting MMLU ====="

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${HF_CKPT_PATH} \
    --tasks mmlu

echo "==== Starting Passkey ====="

mkdir -p ${AMLT_OUTPUT_JOB_PATH}

python ./scripts_passkey/eval_passkey_ac.py \
    --model_path ${HF_CKPT_PATH} \
    --output_dir ${AMLT_OUTPUT_JOB_PATH}

echo "Done."