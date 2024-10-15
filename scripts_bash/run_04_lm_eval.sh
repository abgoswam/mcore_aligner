#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

if [ -z "$JOB_PATH" ]; then
  JOB_PATH=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced/mistral_ckpts
fi

model_path=${JOB_PATH}/Mistral-7B-v0.1-to-HF1

output_path=.

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu \