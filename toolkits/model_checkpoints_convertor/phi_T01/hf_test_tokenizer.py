from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer
tokenizer_path = "./temp2"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# Encode the input prompt and move it to the GPU
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

# Print the input IDs and their corresponding tokens
input_ids = inputs["input_ids"][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print("Input IDs:", input_ids)
print("Tokens:", tokens)
