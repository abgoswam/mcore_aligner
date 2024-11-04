set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)


MEGATRON_PATH=AMA-Megatron-LM-10152024
cd ${MEGATRON_PATH}

set -u

# python tools/preprocess_data.py \
#        --input /mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/datasets_test/mistral-datasets/output_dir_ultrachat_200k/text_token_count_dataset.jsonl \
#        --output-prefix my_mistral_ultrachat_200k \
#        --tokenizer-type MistralTokenizer \
#        --tokenizer-model /mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1/tokenizer.model \
#        --workers 32

python tools/preprocess_data.py \
       --input /mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/datasets_test/mistral-datasets/output_dir_ultrachat_200k/text_token_count_dataset.jsonl \
       --output-prefix my_mistral_ultrachat_200k \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model mistralai/Mistral-7B-v0.1 \
       --workers 32