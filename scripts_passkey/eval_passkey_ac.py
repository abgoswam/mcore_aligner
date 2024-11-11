# Different kinds of HF quantization summarized here
# https://github.com/huggingface/transformers/blob/main/docs/source/en/kv_cache.md
# replace lines 68-76 accordingly

from numpy import random
import argparse
import re
import json
import os
# import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import time
import traceback

if not torch.cuda.is_available():
    raise ValueError("This script requires a GPU to run.")

def generate_prompt_landmark(max_tokens, seed, prefix_fraction): #(n_garbage, seed, n_garbage_prefix):
    """Generates a text file and inserts an passkey at a random position."""
    n_garbage = (max_tokens - 32 - 26 - 11) // 25
    n_garbage_prefix = int(prefix_fraction * n_garbage)

    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    #garbage_inf = " ".join([garbage] * 5000)
    #assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage * n_garbage_prefix
    garbage_suffix = garbage * n_garbage_suffix
    #garbage_prefix = garbage_inf[:n_garbage_prefix]
    #garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), pass_key


def extract_int(response):
    # Search for the first sequence of digits in the response
    match = re.search(r'\d+', response)
    # Return the integer if a match is found, otherwise return None
    return int(match.group()) if match else None

def resilient_generate(model, *args, **kwargs):
    oom = False
    print(args)
    print(kwargs)

    try:
        return model.generate(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        print(e)
        print("retrying with cache_implementation='quantized'")
        oom = True
    if oom:
        torch.cuda.empty_cache()
        kwargs["cache_implementation"] = "quantized"
        kwargs["cache_config"] = {"nbits": 4, "backend": "quanto"}
        return model.generate(*args, **kwargs)

def model_gen(model, tokenizer, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    kwargs = { "max_new_tokens" : 20 }
    outputs = resilient_generate(model, **inputs, **kwargs)

    prompt_text_token_len = inputs['input_ids'].shape[1]
    generated_text = tokenizer.decode(outputs[0][prompt_text_token_len:], skip_special_tokens=True)
    
    return generated_text, prompt_text_token_len


def get_pass_key(model, tokenizer, prompt_text):
    generated_text, prompt_text_token_len = model_gen(model, tokenizer, prompt_text)

    pass_key = extract_int(generated_text)
    if pass_key is None:
        pass_key = -1
    return pass_key, prompt_text_token_len


def plot_heatmap(results, vmax, max_tokens, output_dir):
    df = pd.DataFrame(results)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        vmin=0,  # Set the colormap minimum to 0
        vmax=vmax, # Set the colormap maximum to the maximum possible score  
        cbar_kws={'label': 'Score'}
    )

    # More aesthetics
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    plt.savefig(os.path.join(output_dir, f"heatmap_{max_tokens}.png"))


def estimate_passkey_retrieval_accuracy(model, tokenizer, trials, context_size):
    avg_tokens = None
    results = []
    for prefix_numerator in range(0, 11):
        correct_cnt = 0
        total_tokens = 0
        for k in range(trials):
            prompt_text, pass_key = generate_prompt_landmark(context_size, k, prefix_numerator/10.0)
            pred, length = get_pass_key(model, tokenizer, prompt_text)
            correct_cnt += 1 if pred == pass_key else 0
            total_tokens += length
        accuracy = correct_cnt/trials
        avg_tokens = total_tokens//trials if avg_tokens is None else avg_tokens
        depth = prefix_numerator/10.0
        print(f"token length {avg_tokens}, depth {depth}, accuracy {accuracy}")
        results.append({"Context Length": avg_tokens, "Document Depth": round(depth*100, -1), "Score": correct_cnt})

    total_correct = [result["Score"] for result in results]
    length_accu = []
    length_accu.append({"Context Length": context_size, "Accuracy": float(sum(total_correct))/(len(results)*trials)})
    return results, length_accu


def str2bool(v):
    yes = {'yes', 'true', 't', 'y', '1'}
    no = {'no', 'false', 'f', 'n', '0'}
    if v.lower() in yes:
        return True
    elif v.lower() in no:
        return False
    else:
        raise argparse.ArgumentTypeError(f'Expected one of {yes} or {no}, but got {v}.')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2**18)
    parser.add_argument("--max_position_embeddings", type=int, default=2**18)
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B")

    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=98052)
    parser.add_argument("--use_flash_attention", type=str, default="true", help="Whether to use flash attention or not.")
    parser.add_argument("--output_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))

    args, unknown_args = parser.parse_known_args()
    print(f"known_args: {args}")
    print(f"unknown_args: {unknown_args}")

    random.seed(args.seed)

    checkpoint = args.model_path
    for i in range(20):
        try:
            config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True) 
        except PermissionError:
            print(f"PermissionError while trying to load config from {checkpoint}. Retrying...")
            time.sleep(1)
            continue
        break
    else:
        raise PermissionError(f"Failed to load config from {checkpoint}")

    print(f"Pre config: {config}")

    if args.max_position_embeddings > 0:
        print(f"Updating config.max_position_embeddings from {config.max_position_embeddings} to {args.max_position_embeddings}")
        config.max_position_embeddings = args.max_position_embeddings

    # config.rope_scaling = {
    #     "long_factor": [1.0800000429153442, 1.1100000143051147, 1.1399999856948853, 1.340000033378601, 1.5899999141693115, 1.600000023841858, 1.6200000047683716, 2.620000123977661, 3.2300000190734863, 3.2300000190734863, 4.789999961853027, 7.400000095367432, 7.700000286102295, 9.09000015258789, 12.199999809265137, 17.670000076293945, 24.46000099182129, 28.57000160217285, 30.420001983642578, 30.840002059936523, 32.590003967285156, 32.93000411987305, 42.320003509521484, 44.96000289916992, 50.340003967285156, 50.45000457763672, 57.55000305175781, 57.93000411987305, 58.21000289916992, 60.1400032043457, 62.61000442504883, 62.62000274658203, 62.71000289916992, 63.1400032043457, 63.1400032043457, 63.77000427246094, 63.93000411987305, 63.96000289916992, 63.970001220703125, 64.02999877929688, 64.06999969482422, 64.08000183105469, 64.12000274658203, 64.41000366210938, 64.4800033569336, 64.51000213623047, 64.52999877929688, 64.83999633789062],
    #     "short_factor": [1.0, 1.0199999809265137, 1.0299999713897705, 1.0299999713897705, 1.0499999523162842, 1.0499999523162842, 1.0499999523162842, 1.0499999523162842, 1.0499999523162842, 1.0699999332427979, 1.0999999046325684, 1.1099998950958252, 1.1599998474121094, 1.1599998474121094, 1.1699998378753662, 1.2899998426437378, 1.339999794960022, 1.679999828338623, 1.7899998426437378, 1.8199998140335083, 1.8499997854232788, 1.8799997568130493, 1.9099997282028198, 1.9399996995925903, 1.9899996519088745, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0799996852874756, 2.0899996757507324, 2.189999580383301, 2.2199995517730713, 2.5899994373321533, 2.729999542236328, 2.749999523162842, 2.8399994373321533],
    #     "type": "longrope"
    # }
    print(f"Post config: {config}")

    torch_dtype = config.torch_dtype if config.torch_dtype in [torch.float16, torch.bfloat16] else torch.bfloat16 # force run bfloat16
    if str2bool(args.use_flash_attention):
        # torch_dtype = config.torch_dtype if config.torch_dtype in [torch.float16, torch.bfloat16] else torch.bfloat16 # force run bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            config=config, 
            device_map="auto",
            trust_remote_code=True, 
            torch_dtype=torch_dtype, 
            attn_implementation="flash_attention_2")

    # # torchtype consistent
    # ds_engine = deepspeed.init_inference(model, tensor_parallel={"tp_size": world_size}, dtype=torch_dtype, replace_with_kernel_inject=False) #todo: would be nice to have replace_with_kernel_inject=True but it gives cuda compilation errors
    # model = ds_engine.module

    # Move the model to GPUs (if not automatically done)
    # model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"tokenizer.pad_token: {tokenizer.pad_token}")
    if model.config.pad_token_id is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id
    print(f"model.config.pad_token_id: {model.config.pad_token_id}")
    if model.generation_config.pad_token_id is None and model.generation_config.eos_token_id is not None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    prompt_text = "Tell me a Fun fact:"
    generated_text, prompt_text_token_len = model_gen(model, tokenizer, prompt_text)
    print(f"generated_text: {generated_text}")

    result_list = list()
    accu_list = list()
    length_list = [2**i for i in range(11,22) if 2**i <= args.max_length]
    for context_size in reversed(length_list):
        try:
            accuracies, length_accu = estimate_passkey_retrieval_accuracy(model, tokenizer, args.trials, context_size)
            result_list.extend(accuracies) # append({"context_size": context_size, "accuracy": accuracy})
            accu_list.extend(length_accu)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                # Handle out of memory error gracefully
                print(f"CUDA out of memory. context_size: {context_size}")
                traceback.print_exc()
                # Additional actions like freeing up memory or reducing batch size can be taken here
            else:
                # Re-raise the exception if it's not related to CUDA out of memory
                raise

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir,"results.jsonl")
        with open(output_file, "w") as out:
            for result in result_list:
                out.write(json.dumps(result) + "\n")
        # calculate length-wise accuracy
        output_accu_file = os.path.join(args.output_dir,"accuracy.jsonl")
        with open(output_accu_file, "w") as out:
            for accu in accu_list:
                out.write(json.dumps(accu) + "\n")
        plot_heatmap(result_list, args.trials, args.max_length, args.output_dir)


if __name__ == "__main__":
    print("Starting eval passkey")
    main()
    print("Done")