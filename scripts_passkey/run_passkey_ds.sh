#!/usr/bin/env bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

# deepspeed 
#     eval_passkey.py
#     --use_flash_attention ${{inputs.use_flash_attention}} 
#     --max_length ${{inputs.max_length}} 
#     --trials ${{inputs.trials}} 
#     --rope_scaling_factor ${{inputs.rope_scaling_factor}} 
#     --rope_scaling_type ${{inputs.rope_scaling_type}} 
#     --seed ${{inputs.seed}} 
#     --model_path ${{inputs.model_path}} 
#     --passkey_phrase ${{inputs.passkey_phrase}} 
#     --output_dir ${{outputs.output_dir}}

# [2024-10-10 12:03:02,561] [INFO] [runner.py:568:main] cmd = /opt/miniconda/bin/python3.10 -u -m deepspeed.launcher.launch --world_info=eyJub2RlLTAiOiBbMCwgMSwgMiwgMywgNCwgNSwgNiwgN119 --master_addr=10.6.69.171 --master_port=29500 --enable_each_rank_log=None eval_passkey.py --use_flash_attention True --max_length 262144 --trials 5 --rope_scaling_factor 1.0 --rope_scaling_type linear --seed 98052 --model_path /scratch/azureml/cr/j/5ddef37198784ab08414e6053a76dff1/cap/data-capability/wd/INPUT_model_path --passkey_phrase long context --output_dir /scratch/azureml/cr/j/5ddef37198784ab08414e6053a76dff1/cap/data-capability/wd/output_dir

# [2024-10-10 12:03:02,561] [INFO] [runner.py:568:main] cmd = /opt/miniconda/bin/python3.10 -u -m deepspeed.launcher.launch \
    # --world_info=eyJub2RlLTAiOiBbMCwgMSwgMiwgMywgNCwgNSwgNiwgN119 \
    # --master_addr=10.6.69.171 \
    # --master_port=29500 \
    # --enable_each_rank_log=None \
    # eval_passkey.py \
    # --use_flash_attention True \
    # --max_length 262144 \
    # --trials 5 \
    # --rope_scaling_factor 1.0 \
    # --rope_scaling_type linear \
    # --seed 98052 \
    # --model_path /scratch/azureml/cr/j/5ddef37198784ab08414e6053a76dff1/cap/data-capability/wd/INPUT_model_path \
    # --passkey_phrase long context \
    # --output_dir /scratch/azureml/cr/j/5ddef37198784ab08414e6053a76dff1/cap/data-capability/wd/output_dir

deepspeed /home/aiscuser/mcore_aligner/scripts_passkey/eval_passkey_ds.py \
    --use_flash_attention True \
    --max_length 262144 \
    --trials 5 \
    --rope_scaling_factor 1.0 \
    --rope_scaling_type linear \
    --seed 98052 \
    --model_path  /mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_converted/phi35_pretrained/gilopez_Phi-3_1-mling \
    --passkey_phrase long context \
    --output_dir /home/aiscuser/mcore_aligner/output_dir_passkey