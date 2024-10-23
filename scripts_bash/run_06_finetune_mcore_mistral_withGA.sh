#!/bin/bash
if [ -z "$JOB_PATH" ]; then
  JOB_PATH=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced
  MCORE_ALIGNER_PATH=/home/aiscuser/mcore_aligner
else
  # this is amulet. so JOB_PATH will be set, and code has been uploaded. 
  MCORE_ALIGNER_PATH=.
fi

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

cd ${MCORE_ALIGNER_PATH}/examples/mistral &&
ls -lh &&
sh run_finetune_mcore_mistral_withGA.sh  \
    dsw  \
    ../../ \
    7B   \
    1    \
    8 \
    1e-5   \
    1e-6   \
    8192  \
    128  \
    0   \
    bf16  \
    1   \
    1  \
    sel  \
    true   \
    false  \
    false   \
    false   \
    false \
    500  \
    /mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/datasets_test/mistral-datasets/alpaca_zh-mistral-train.json   \
    /mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/datasets_test/mistral-datasets/alpaca_zh-mistral-valid.json   \
    /mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced/Mistral-7B-v0.1-to-mcore-tp1-pp1   \
    2000   \
    10   \
    ${JOB_PATH}/output_mcore_mistral_sft5


# ENV=$1                          # Running environment: dlc, dsw
# MEGATRON_PATCH_PATH=$2          # Path to Megatron Patch code
# MODEL_SIZE=$3                   # Model size: 7B, 13B
# BATCH_SIZE=$4                   # Per GPU batch size: 4, 8
# GLOBAL_BATCH_SIZE=$5            # Global batch size
# LR=$6                           # Learning rate: 1e-5, 5e-5
# MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
# SEQ_LEN=$8                      # Sequence length ==================================
# PAD_LEN=$9                      # Padding length: 100
# EXTRA_VOCAB_SIZE=${10}          # Vocabulary expansion size
# PR=${11}                        # Precision: fp16, bf16
# TP=${12}                        # Tensor parallelism =================================
# PP=${13}                        # Pipeline parallelism
# AC=${14}                        # Activation checkpointing mode: sel, full
# DO=${15}                        # Use Megatron's Zero-1 memory optimizer: true, false
# FL=${16}                        # Use Flash Attention: true, false
# SP=${17}                        # Use sequence parallelism: true, false
# TE=${18}                        # Use Transformer Engine: true, false
# MOE=${19}                       # Enable MoE: true, false
# SAVE_INTERVAL=${20}             # Checkpoint save interval ===========================
# DATASET_PATH=${21}              # Training dataset path (in jsonl) ===================
# VALID_DATASET_PATH=${22}        # Validation dataset path (in jsonl) =================
# PRETRAIN_CHECKPOINT_PATH=${23}  # Pre-trained model path =============================
# TRAIN_ITERS=${24}               # Number of training steps
# LR_WARMUP_ITERS=${25}           # Number of warmup steps
# OUTPUT_BASEPATH=${26}           # Output path for training ===========================