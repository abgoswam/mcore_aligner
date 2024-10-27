from transformers import AutoConfig, AutoModelForCausalLM

# Load the configuration
print("Initializing config...")
config = AutoConfig.from_pretrained("/home/aiscuser/mcore_aligner/toolkits/model_checkpoints_convertor/phi_T01/config.json")

# Initialize the model with random weights
print("Initializing model...")
model = AutoModelForCausalLM.from_config(config)

# Save the model and config
print("Saving...")
# model.save_pretrained("/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced/phi_T01_ckpts/hf_init")
model.save_pretrained("temp")

print("done")