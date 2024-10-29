from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load the tokenizer
# tokenizer_path = "./temp2"

# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# # Encode the input prompt and move it to the GPU
# prompt = "Once upon a time"
# inputs = tokenizer(prompt, return_tensors="pt")

# # Print the input IDs and their corresponding tokens
# input_ids = inputs["input_ids"][0].tolist()
# tokens = tokenizer.convert_ids_to_tokens(input_ids)

# print("Input IDs:", input_ids)
# print("Tokens:", tokens)

import tiktoken

name = "o200k_base"
o200k_base = tiktoken.get_encoding(name)

extra_special_tokens = {
    "<|im_start|>": 200016,  # ChatML support
    "<|im_end|>": 200017,  # ChatML support
}

enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name=f'{name}_im',
    pat_str=o200k_base._pat_str,
    mergeable_ranks=o200k_base._mergeable_ranks,
    special_tokens={
        **o200k_base._special_tokens,
        **extra_special_tokens,
    }
)

# Encode the string to get token IDs
token_ids = enc.encode("hello world")
print("Token IDs:", token_ids)

# Decode token IDs back to strings
tokens = [enc.decode_single_token_bytes(token_id).decode("utf-8", errors="replace") for token_id in token_ids]
print("Tokens:", tokens)

# Print the vocabulary size
vocab_size = enc.n_vocab
print("Vocabulary Size:", vocab_size)

print("Special Tokens")
print(o200k_base._special_tokens)

print("Special Tokens")
print(enc._special_tokens)

# _eod_id = enc.encode("<|endoftext|>", allowed_special="all")[0] # 199999
# _vocab_size = enc.n_vocab
# _unk_id = _eod_id
# _bos_id = _eod_id
# _eos_id = _eod_id

# print(f"{_eod_id}:{_vocab_size}:{_unk_id}:{_bos_id}:{_eos_id}")

# tokens = [enc.decode_single_token_bytes(token_id).decode("utf-8", errors="replace") for token_id in [199999]]
# print("Tokens:", tokens)

tokenizer = enc
print(f"{tokenizer.eot_token}")

print("done")