from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
# model_name = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1/"
# model_name = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced/phi_T01_ckpts/2024-09-13-phi3min-tp1pp1-1800b-HF"
# model_name = "/home/aiscuser/mcore_aligner/temp"
# model_name = "/home/aiscuser/mcore_aligner/temp_hf_init_ckpt_2"
model_name = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/phi3_ckpts/gilopez_Phi-3_1-mling/"

model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True, trust_remote_code=True)

# Move the model to the GPU
device = "cuda"  # or "cuda:0" if you have multiple GPUs and want to specify the first one
model.to(device)
print(model)

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# # Encode the input prompt and move it to the GPU
# prompt = "Once upon a time"
# inputs = tokenizer(prompt, return_tensors="pt").to(device)

# # Generate text
# output = model.generate(**inputs, max_length=100, do_sample=True)

# # Decode the output and print it
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)

# # Get the vocabulary size
# vocab_size = tokenizer.vocab_size
# print("Vocabulary size:", vocab_size)

print("done")
