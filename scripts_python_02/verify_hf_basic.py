import debugpy
debugpy.listen(5678)  # 5678 is port
print("Waiting for debugger attach")
debugpy.wait_for_client()
debugpy.breakpoint()
print('break on this line')

"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from megatron.training import get_args, get_tokenizer
from megatron.training import print_rank_0
from megatron.core import mpu
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.core.models.gpt import GPTModel
from megatron.training import get_model
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.inference.text_generation import generate_and_post_process
from megatron.inference.text_generation import beam_search_and_post_process
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

import torch
from typing import Union
import megatron
from pprint import pprint

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the core GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')

    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=False,
            pre_process=pre_process,
            post_process=post_process
        )
    else:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling
        )

    return model

def verify_tokenizer(hf_tokenizer, prompt="Fun fact:"):
    mg_tokenizer = get_tokenizer()

    assert len(hf_tokenizer) == mg_tokenizer.vocab_size

    hf_x_list = hf_tokenizer.encode(prompt)
    mg_x_list = mg_tokenizer.tokenize(prompt)
    assert hf_x_list == mg_x_list, f">> {hf_x_list=}\n{mg_x_list=}"


def verify_logits(
            hf_model,
            hf_tokenizer, 
            mg_model, 
            prompt="Fun fact:", 
            verify=True):
    
    # first verify tokenizer.
    verify_tokenizer(hf_tokenizer)

    # Extract logits (1)
    inputs = hf_tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = hf_model(**inputs)
    
    print("inputs:{}".format(inputs))
    print("output.logits:{}".format(output.logits))

    inputs['position_ids']=inputs['attention_mask'] # otherwise we hit error.
    mg_logits0 = mg_model(**inputs)
    print("mg_logits0:{}".format(mg_logits0))

    if verify:
        print("Doing verification of logits....")
        # TODO (agoswami): Not sure how this works for Turing ckpoint. Need to investigate
        # hf_logits = hf_model(hf_x).logits.view(hf_x.size(-1), -1).float()
        # mg_logits = mg_model(hf_x).view(hf_x.size(-1), -1).float()
        # We will use what we know.
        hf_logits = output.logits
        mg_logits = mg_logits0

        # remove mg padding
        mg_logits = mg_logits[:, :hf_logits.size(1)].float()
        max_diff = (hf_logits - mg_logits).abs().max().item()
        q99_diff = torch.quantile((hf_logits - mg_logits).abs(), 0.99).item()
        q95_diff = torch.quantile((hf_logits - mg_logits).abs(), 0.95).item()

        print(f'> {hf_logits=}\n> {mg_logits=}')
        print(f'> {max_diff=}, {q99_diff=}, {q95_diff=}')

        assert torch.allclose(hf_logits, mg_logits, rtol=1e-1, atol=3e-1), \
            f'> {hf_logits=}\n> {mg_logits=}'

        hf_argmax = hf_logits.argmax(-1)
        mg_argmax = mg_logits.argmax(-1)
        assert (hf_argmax == mg_argmax).all(), f'> {hf_argmax=}, {mg_argmax=}'
        # assert q99_diff < 0.1
        # assert q95_diff < 0.1
        print('> Verification passed.')
        # Uncomment the following the ipdb to do interactive debug
        # import ipdb; ipdb.set_trace()
    else:
        print("Skipping verification of logits....")


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys

    initialize_megatron()
    args = get_args()
    pprint(vars(args))

    #########################################################
    # Inspect provided MG ckpt.

    base_mcore_ckpt_path = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_converted/mistral_ckpts/Mistral-7B-v0.1-to-mcore-tp1-pp1/"

    # Load the checkpoint
    checkpoint = torch.load(
                    os.path.join(base_mcore_ckpt_path, "release/mp_rank_00/model_optim_rng.pt"), 
                    map_location=torch.device('cpu'))
    # print(checkpoint)

    # Print out args of the base ckpt.
    print(checkpoint['args'])

    # # Access the state dictionary
    model_state_dict = checkpoint['model']

    # # Print the layers and their sizes
    # # Iterate over the state_dict items and print layer names and their sizes
    for layer_name, tensor in model_state_dict.items():
        if isinstance(tensor, torch.Tensor):  # Check if the item is a tensor
            print(f"Layer: {layer_name}, Size: {tuple(tensor.size())}")
        else:
            # print(f"Layer: {layer_name} is not a tensor, found type: {type(tensor)}")
            pass

    ########################################################
    # create_huggingface_model
    
    hf_path = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1"
    hf_tok = AutoTokenizer.from_pretrained(hf_path)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hf_model = hf_model.to(device).eval()
    print(hf_model)

    # Print out the parameter names and their sizes
    print("[hf_model] Parameters and their sizes:")
    for name, param in hf_model.named_parameters():
        print(f"{name}: {param.size()}")


    #########################################################
    # create_megatron_model

    mg_model_list = get_model(model_provider, wrap_with_ddp=False)

    # Print out args of the init ckpt.
    # TODO (agoswami)

    # MG Print out the parameter names and their sizes
    for name, param in mg_model_list[0].named_parameters():
        print(f"{name}: {param.size()}")

    verify_logits( 
        hf_model=hf_model, 
        hf_tokenizer=hf_tok,
        mg_model=mg_model_list[0],
        verify=False
    )

    ########################################################
    # load_megatron_model

    args.load = base_mcore_ckpt_path
    _ = load_checkpoint(mg_model_list, None, None)

    assert len(mg_model_list) == 1, "Above condition should have caught this"
    mg_model = mg_model_list[0].eval()

    #######################################################
    # verify_logits

    verify_logits(
        hf_model=hf_model, 
        hf_tokenizer=hf_tok,
        mg_model=mg_model,
        verify=True
    )

    #######################################################
    # Generation based.

    tokens_to_generate = 100
    logprobs = True
    top_k = 1
    top_p = 0
    top_p_decay = 0
    top_p_bound = 0 
    temperature = 1
    add_BOS = False
    stop_on_double_eol = False
    stop_on_eol = False
    prevent_newline_after_colon = False
    random_seed = -1

    response, response_seg, response_logprobs, _ = \
        generate_and_post_process(
        mg_model,
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        return_output_log_probs=logprobs,
        top_k_sampling=top_k,
        top_p_sampling=top_p,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        add_BOS=add_BOS,
        use_eod_token_for_early_termination=True,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        random_seed=random_seed)


    hf_x = hf_tok.encode(prompts[0], return_tensors='pt').cuda()
    hf_y = hf_model.generate(hf_x, top_k=1)
    hf_response = hf_tok.decode(hf_y[0])

    print(f'HF: prompt: {hf_tok.encode(prompts[0])}')
    print(f'MG: prompt: {mg_tokenizer.tokenize(prompts[0])}')
    print(f'HF: {hf_y[0].tolist()}')
    print(f'MG: {mg_tokenizer.tokenize(response)[0]}')

    assert hf_response == response[0], f"Mismatch detected:\nHF Response: {hf_response}\nMG Response: {response}"

    hf_logits = hf_model(hf_y).logits.squeeze(0)
    hf_logits = hf_logits.gather(1, hf_x[0, :, None])
    
    print((mg_model.module.embedding.word_embeddings.weight[:32000] - hf_model.model.embed_tokens.weight).abs().max())
    print((mg_model.module.decoder.layers[0]))


    mg = mg_model.module.decoder.layers[0]
    hf = hf_model.model.layers[0]

    x = torch.randn(1, 8, 3072, dtype=torch.bfloat16).cuda()

    # y_mg = mg(x, attention_mask=None)
    # y_hf = hf(x)
    

    # model.module.decoder.layers[0].self_attention.linear_proj.weight

    all_hooks = []
    def get_hook(inter_dict):
        def hook(mod, input, output):
            inter_dict['input'] = input
            inter_dict['output'] = output
        return hook

    def register_hook(mod, inter_dict):
        for k, v in mod._modules.items():
            _dict = inter_dict.get(k, {})
            inter_dict[k] = _dict
            register_hook(v, _dict)
        hook = mod.register_forward_hook(get_hook(inter_dict))
        all_hooks.append(hook)


    def remove_hooks():
        for hook in all_hooks:
            hook.remove()

    hf_all = {}; register_hook(hf, hf_all)
    mg_all = {}; register_hook(mg, mg_all)

    with torch.no_grad():
        hf_model(hf_x)
        mg_model(hf_x, None, None)




    z = mg.self_attention.get_query_key_value_tensors(mg_all['input_layernorm']['input'][0])
    hf.self_attn.k_proj.bias
    z[1]
    import ipdb; ipdb.set_trace()


    z_hf = hf.mlp(hf.post_attention_layernorm(x)) + x
    _y, _b = mg.mlp(x)
    z_mg = _y + _b + x

    mg.self_attention(x) # layernorm -> qkv -> flash -> project

    hf.self_attn(hf.input_layernorm(x))






