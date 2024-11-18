
cd ./toolkits/model_checkpoints_convertor/phi35_pretrained &&
sh hf2mcore_convertor.sh \
3B \
/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_converted/phi35_pretrained/gilopez_Phi-3_1-mling \
../../../     \
/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_converted/phi35_pretrained/gilopez_Phi-3_1-mling \
/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_converted/phi35_pretrained/phi35_pretrained_mcore_tp1_pp1  \
1  \
1  \
64  \
0  \
0  \
0 \
false