# import debugpy
# debugpy.listen(5678)  # 5678 is port
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')

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
            rotary_percent=args.rotary_percent
        )

    return model




if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys

    initialize_megatron()
    args = get_args()
    pprint(vars(args))

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    print(model)

    load_path = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_converted/mistral_ckpts/Mistral-7B-v0.1-to-mcore-tp1-pp1/"
    args.load = load_path
    _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0].eval()
    tokenizer = get_tokenizer()

    hf_path = "/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_base/mistral_ckpts/Mistral-7B-v0.1"
    # sys.path.append(hf_path)
    # from modeling_mistral import MistralForCausalLM
    hf_tok = AutoTokenizer.from_pretrained(hf_path)
    # hf_model = MistralForCausalLM.from_pretrained(hf_path)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
    hf_model.cuda().eval().to(torch.bfloat16)

    prompts = ["This is a test of "]
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
        model,
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

    print(f'HF: prompt: {hf_tok.encode(prompts[0])}')
    print(f'MG: prompt: {tokenizer.tokenize(prompts[0])}')
    print(f'HF: {hf_y[0].tolist()}')
    print(f'MG: {tokenizer.tokenize(response)[0]}')

    hf_logits = hf_model(hf_y).logits.squeeze(0)
    hf_logits = hf_logits.gather(1, hf_x[0, :, None])
    
    print((model.module.embedding.word_embeddings.weight[:32000] - hf_model.model.embed_tokens.weight).abs().max())
    print((model.module.decoder.layers[0]))


    mg = model.module.decoder.layers[0]
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
        model(hf_x, None, None)




    z = mg.self_attention.get_query_key_value_tensors(mg_all['input_layernorm']['input'][0])
    hf.self_attn.k_proj.bias
    z[1]
    import ipdb; ipdb.set_trace()


    z_hf = hf.mlp(hf.post_attention_layernorm(x)) + x
    _y, _b = mg.mlp(x)
    z_mg = _y + _b + x

    mg.self_attention(x) # layernorm -> qkv -> flash -> project

    hf.self_attn(hf.input_layernorm(x))






