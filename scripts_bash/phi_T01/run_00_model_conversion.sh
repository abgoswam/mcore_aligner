#!/usr/bin/env bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

# echo "***** Step 1: convert weights HF to MCore.***** "
# bash ./scripts_bash/run_02_mistral_hf2mcore.sh
# if [ $? -ne 0 ]; then
#   echo "Error in run_02_mistral_hf2mcore.sh"
#   exit 1
# fi

echo "***** Step 2: convert MCore to HF***** "
bash ./scripts_bash/phi_T01/run_03_phi_mcore2hf.sh
if [ $? -ne 0 ]; then
  echo "Error in run_03_phi_mcore2hf.sh"
  exit 1
fi

# echo "***** Step 3: LM_Eval ***** "
# bash ./scripts_bash/run_04_lm_eval.sh
# if [ $? -ne 0 ]; then
#   echo "Error in run_04_lm_eval.sh"
#   exit 1
# fi