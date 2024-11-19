from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Mistral model and tokenizer
# model_name = "mistralai/Mistral-7B-v0.1"  # Replace with your specific Mistral model name
model_name = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1"  # Replace with your specific Mistral model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # Move model to GPU

# Input text
input_text = "Fun fact:"

# Tokenize input with attention mask
inputs = tokenizer(input_text, return_tensors="pt").to(device)  # Move inputs to GPU
print(inputs)

print("==" * 30)
# Forward pass to get logits
with torch.no_grad():
    output = model(**inputs)

# Extract logits
logits = output.logits
print(logits.shape)
print(logits)

print("==" * 30)
# Use the generate method for proper token generation
generated_ids = model.generate(**inputs, max_length=50)  # Adjust max_length as needed
tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
print(generated_ids)
print(tokens)

# Forward pass to get logits
with torch.no_grad():
    output = model(generated_ids)

# Extract logits
logits = output.logits
print(logits.shape)
print(logits)

# Decode tokens and display logits
generated_ids = torch.argmax(logits, dim=-1)
tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
print(generated_ids)
print(tokens)

print("==" * 30)
# Use the generate method for proper token generation
generated_ids = model.generate(**inputs, top_k=1)  # Adjust max_length as needed
tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
print(generated_ids)
print(tokens)

# Forward pass to get logits
with torch.no_grad():
    output = model(generated_ids)

# Extract logits
logits = output.logits
print(logits.shape)
print(logits)

print("Done")
