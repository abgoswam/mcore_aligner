# if [ -z "$JOB_PATH" ]; then
#   JOB_PATH=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced
# fi

cd ./toolkits/model_checkpoints_convertor/mistral &&
sh hf2mcore_convertor.sh \
7B \
/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1 \
../../../     \
/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/jobs_test/massive-man/output_mcore_mistral_cpt2/checkpoint/dsw-pretrain-megatron-gpt3-7B-lr-1e-5-bs-1-seqlen-4096-pr-bf16-tp-1-pp-1-ac-sel-do-true-sp-false-tt-280000000-wt-10000 \
/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/jobs_test/massive-man/output_mcore_mistral_cpt2/ckpt-8400-HF  \
1  \
1  \
0  \
0  \
0  \
0 \
true