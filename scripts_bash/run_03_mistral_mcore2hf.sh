if [ -z "$JOB_PATH" ]; then
  JOB_PATH=/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced
fi

cd ./toolkits/model_checkpoints_convertor/mistral &&
sh hf2mcore_convertor.sh \
7B \
/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1 \
../../../     \
${JOB_PATH}/Mistral-7B-v0.1-to-mcore-tp1-pp1 \
${JOB_PATH}/Mistral-7B-v0.1-to-HF1  \
1  \
1  \
0  \
0  \
0  \
0 \
true