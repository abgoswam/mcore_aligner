import json
import os
import tiktoken
from typing import Collection, List, Optional, Dict, Set, Tuple, Union
import base64

from transformers import PreTrainedTokenizer, AddedToken, AutoConfig
from transformers.models.auto.tokenization_auto import get_tokenizer_config
import tiktoken

"""
    This tokenizer is almost identical to tiktoken.get_encoding("cl100k_base")
    with a few additional special tokens to support the ChatML format.

    TODO(agoswami): Right now, it does not save the special tokens to the vocab file.
    Maybe in the future, that would be useful? Can add that support later.

"""

def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }

SPECIAL_TOKENS = {
    # tiktoken.get_encoding("o200k_base")._special_tokens
    '<|endoftext|>': 199999, 
    # Special tokens added into megatron. https://github.com/microsoft/Megatron-LM/blob/d158a0851ee9904505825ee156c390d74c96e956/megatron/training/tokenizer/tokenizer.py#L835
    '<|im_start|>': 200016, 
    '<|im_end|>': 200017,
    # The final end of prompt token
    # (unused, but present as a part of tiktoken.get_encoding("cl100k_base")._special_tokens)
    '<|endofprompt|>': 200018,
}

class PhiT01Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "o200k_base.tiktoken"
    }

    model_input_names: List[str] = ["input_ids", "attention_mask"]
    padding_side = "left"

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        errors: str = "replace",
        **kwargs
    ) -> None:
        # PreTrainedTokenizer's init calls _add_tokens, which in turn checks
        # if the token is present in `self.special_tokens``. Hence instantiating it here.
        # The way Qwen gets around this is by checking against SPECIAL_TOKENS
        # But I think it's better to check against the objects own `special_tokens`
        # in case we eventually want to allow the tokenizer to have special tokens.
        self.special_tokens = SPECIAL_TOKENS

        super().__init__(**kwargs)
        self.errors = errors

        # Initialize tiktoken encoding
        base = tiktoken.get_encoding("o200k_base")
       
        self.mergeable_ranks = base._mergeable_ranks
        self.pat_str = base._pat_str

        # extended 
        enc = tiktoken.Encoding(
            name="phi_t01",
            pat_str=self.pat_str,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc

        self.decoder: Dict[int, bytes] = {
            v: k for k, v in self.mergeable_ranks.items()
        }
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["tokenizer"]
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        enc = tiktoken.Encoding(
            name="o200k_im",
            pat_str=self.pat_str,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc
    
    def __len__(self):
        return self.tokenizer.n_vocab
    
    def _add_tokens(
            self,
            new_tokens: Union[List[str], List[AddedToken]],
            special_tokens: bool = False,
    ) -> int:
        if not special_tokens and new_tokens:
            raise ValueError("Only special tokens can be added to this tokenizer")
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in self.special_tokens:
                raise ValueError(
                    "For now, we do not support unknown special tokens\n"
                    "In the future, if there is a need for this, we can add special tokens to the tokenizer\n"
                    "starting from rank 100261 - 100263 and then 100266 - 100275.\n"
                    "And finally, we can re-construct the enc object back\n"
                )
        return 0
    
    def get_vocab(self) -> Dict[Union[str, bytes], int]:
        return {**self.mergeable_ranks, **self.special_tokens}

# Instantiate your custom tokenizer
hf_tokenizer = PhiT01Tokenizer()

# # Example usage: Encoding and decoding
# tokens = hf_tokenizer.tokenize("Hello, world!")
# print(tokens)
# ids = hf_tokenizer.convert_tokens_to_ids(tokens)
# print(ids)
# text = hf_tokenizer.convert_tokens_to_string(tokens)
# print(text)

# # Example usage: Saving the tokenizer
# hf_tokenizer.save_vocabulary("./tokenizer")


