from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Mistral model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"  # Replace with your specific Mistral model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input text
input_text = "Fun fact:"

# Tokenize input with attention mask
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output with greedy decoding
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,  # Adjust for the desired output length
        do_sample=False
    )

# Get logits for each token in the generated sequence
with torch.no_grad():
    output_logits = model(outputs)

logits = output_logits.logits
print(logits.shape)
print(logits)
print("Done")