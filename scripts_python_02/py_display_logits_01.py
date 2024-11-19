from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Mistral model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"  # Replace with the specific model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input string
input_text = "Fun fact:"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output using greedy decoding
print("Generating output:...")
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=50, 
        do_sample=False  # Greedy decoding
    )

# Decode and print the generated text
print("Decoding output:...")
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# Decode tokens
generated_ids = outputs[0]
tokens = tokenizer.convert_ids_to_tokens(generated_ids)
logits = model(**inputs).logits

# Display each token, its ID, and logits
print(f"{'Token':<20} {'Token ID':<10} {'Logits':<10}")
print("=" * 50)
for token, token_id, logit in zip(tokens, generated_ids.tolist(), logits[0]):
    print(f"{token:<20} {token_id:<10} {logit.tolist()[:5]}")  # Show top 5 logits for brevity
