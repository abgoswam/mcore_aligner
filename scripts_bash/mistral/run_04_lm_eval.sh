#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)

if [ -z "$JOB_PATH" ]; then
  JOB_PATH=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced/mistral_ckpts
fi

set -u

# model_path=${JOB_PATH}/Mistral-7B-v0.1-to-HF-2
# model_path=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/jobs_test/massive-man/output_mcore_mistral_cpt2/ckpt-1400-HF
# model_path=/mnt/syntheticpipelinetrainerv1/jobs_test/stable-mouse/out_hf_ckpt
# model_path=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1
model_path=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_converted/mistral_ckpts/Mistral-7B-v0.1-HF-01


output_path=.

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu \