
from transformers import AutoTokenizer

# Load the Mistral tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Tokenize the text
text = "hello world"
tokens = tokenizer(text, return_tensors="pt")

# Display tokenized output
print("Input IDs:", tokens["input_ids"])
print("Attention Mask:", tokens["attention_mask"])
