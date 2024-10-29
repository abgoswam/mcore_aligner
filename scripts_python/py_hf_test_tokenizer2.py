import tiktoken
from transformers import GPT2TokenizerFast
import json

# Step 1: Define the original tiktoken tokenizer and special tokens
name = "o200k_base"
o200k_base = tiktoken.get_encoding(name)

extra_special_tokens = {
    "<|im_start|>": 200016,  # ChatML support
    "<|im_end|>": 200017,    # ChatML support
}

# Step 2: Prepare the vocabulary and merges for Hugging Face Tokenizer
# Extract vocabulary from tiktoken
vocab = list(o200k_base._mergeable_ranks.keys()) + list(extra_special_tokens.keys())
vocab_dict = {str(token): idx for idx, token in enumerate(vocab)}

# Extract merges: convert the integer pairs to strings using the vocabulary
merge_list = sorted(o200k_base._mergeable_ranks.items(), key=lambda x: x[1])
merges = []

for pair, _ in merge_list:
    # Ensure pair contains exactly two elements before processing
    if len(pair) == 2:
        try:
            token_a = o200k_base.decode([pair[0]])
            token_b = o200k_base.decode([pair[1]])
            merges.append(f"{token_a} {token_b}")
        except Exception as e:
            print(f"Error decoding pair {pair}: {e}")

# Step 3: Save vocabulary and merges to files
with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

with open("merges.txt", "w") as merges_file:
    merges_file.write("\n".join(merges))

# Step 4: Initialize the Hugging Face tokenizer
hf_tokenizer = GPT2TokenizerFast(
    vocab_file="vocab.json",
    merges_file="merges.txt",
    model_max_length=2048,  # Adjust as needed
    padding_side="left"     # Use "right" if preferred
)

# Step 5: Add special tokens
hf_tokenizer.add_special_tokens({'additional_special_tokens': list(extra_special_tokens.keys())})

# Step 6: Save the tokenizer for future use
hf_tokenizer.save_pretrained("path/to/save/hf_tokenizer")

print("Hugging Face tokenizer created and saved successfully!")
