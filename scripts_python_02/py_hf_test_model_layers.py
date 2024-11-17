from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_name = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True, trust_remote_code=True)

# Move the model to the GPU
device = "cuda"  # or "cuda:0" if you have multiple GPUs and want to specify the first one
model.to(device)
print(model)

print("done")
