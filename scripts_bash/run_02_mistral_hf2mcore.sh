cd /home/aiscuser/mcore_aligner/toolkits/model_checkpoints_convertor/mistral &&
sh hf2mcore_convertor.sh \
7B \
/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1 \
../../../     \
/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1 \
/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced/mistral_ckpts/Mistral-7B-v0.1-to-mcore-tp1-pp1  \
1  \
1  \
0  \
0  \
0  \
0 \
false