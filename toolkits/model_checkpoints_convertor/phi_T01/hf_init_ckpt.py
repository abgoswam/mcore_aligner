import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

input_base_path = "/home/aiscuser/mcore_aligner/toolkits/model_checkpoints_convertor/phi_T01/"
output_dir = "temp_hf_init_ckpt"

# Load the configuration
print("Initializing config...")
config = AutoConfig.from_pretrained(os.path.join(input_base_path,"config.json"))

# Initialize the model with random weights
print("Initializing model...")
model = AutoModelForCausalLM.from_config(config)

# Save the model and config
print("Saving...")
model.save_pretrained(output_dir)

# Load the tokenizer from the specified path
tokenizer = AutoTokenizer.from_pretrained(input_base_path, trust_remote_code=True)

# Save the tokenizer to a new path
tokenizer.save_pretrained(output_dir)

print("done")