#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

model_path=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced/phi_T01_ckpts/2024-09-13-phi3min-tp1pp1-1800b-HF-2/

output_path=.

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$model_path,dtype="bfloat16",trust_remote_code=True \
    --tasks mmlu \