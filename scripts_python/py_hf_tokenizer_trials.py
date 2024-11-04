from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_name = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1/"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print(tokenizer)

print(f"{tokenizer.bos_token}:{tokenizer.bos_token_id}")
print(f"{tokenizer.eos_token}:{tokenizer.eos_token_id}")

print("done")