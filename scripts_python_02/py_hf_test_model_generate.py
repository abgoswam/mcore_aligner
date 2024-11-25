import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        # offloaded cache
        kwargs["cache_implementation"] = "offloaded"
        
        # quantized cache
        # kwargs["cache_implementation"] = "quantized"
        # kwargs["cache_config"] = {"nbits": 4, "backend": "quanto"}
        
        return model.generate(*args, **kwargs)

def model_gen(model, tokenizer, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    kwargs = { "max_new_tokens" : 20 }
    outputs = resilient_generate(model, **inputs, **kwargs)

    prompt_text_token_len = inputs['input_ids'].shape[1]
    generated_text = tokenizer.decode(outputs[0][prompt_text_token_len:], skip_special_tokens=True)
    
    return generated_text, prompt_text_token_len

if __name__ == "__main__":
    # Load the Mistral model and tokenizer
    # model_name = "mistralai/Mistral-7B-v0.1"  # Replace with the actual model name on Hugging Face Hub
    model_name = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_converted/phi_T03_ckpts/phi4_with_gqa-tp1pp1-3000b-gbs8388608-mbs2-lr5e-4-HF"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True)

    # Load model onto GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map="auto")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")

    print(f"model.dtype: {model.dtype}")  # Outputs the precision (e.g., torch.float32, torch.bfloat16)
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}, {param.device}")


    # Input prompt
    input_text = "Fun fact 2222 :"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate text
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=100,  # Specify the max length of generated text
        num_beams=5,     # Optional: Adjust for beam search
        no_repeat_ngram_size=2,  # Optional: Prevent repetition
        temperature=0.7, # Optional: Adjust creativity
        top_k=50,        # Optional: Nucleus sampling parameter
        top_p=0.95       # Optional: Nucleus sampling parameter
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)
    print(output_ids)

    generated_text, prompt_text_token_len = model_gen(model, tokenizer, input_text)
    print(generated_text)
