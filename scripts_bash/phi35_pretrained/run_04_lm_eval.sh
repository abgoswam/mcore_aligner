#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

# model_path=${JOB_PATH}/Mistral-7B-v0.1-to-HF-2
model_path=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_converted/phi35_pretrained/gilopez_Phi-3_1-mling
# model_path=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_converted/phi35_pretrained/phi35_pretrained_HF_tp1_pp1

output_path=.

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu \