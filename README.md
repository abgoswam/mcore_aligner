# mcore_aligner

# Model Feature Support

| Feature           | Mistral | Phi_T01 |
|-------------------|:-------:|:-------:|
| **HF → MCore**    | ✔️      | NA      |
| **MCore → HF**    | ✔️      | ✔️      |
| **CPT (idx)**     | ✔️      |         |
| **SFT (idx)**     | ?       |         |
| **SFT (json)**    | ✔️      |         |
| **[P/D]PO**       | ✘       | ✘       |



## Mistral

### Model Conversions.

- HF Mistral -> mcore (Works)
- mcore -> HF Mistral (works)
- lm_eval (MMLU) on converted model: 0.6

 
```bash
amlt run ./submit_amlt_mcore_conversions.yaml
```

### SFT.

In Progress.

### Inference.

#### Batch

- Supports single-gpu / multi-gpu.

```bash
bash ./scripts_bash/run_07_a_inference_batch.sh
```

#### Text Gen Server

```bash
bash ./scripts_bash/run_07_b_inference_text_gen_server.sh
```

```bash
bash tools/text_generation_cli.py localhost:5000
```

# Acknowledgements
This work is built on top of the following papers/repositories:
- [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) at commit [9d3e557](https://github.com/alibaba/Pai-Megatron-Patch/commit/9d3e557b4d5f386a456a49da23aa47af737baaf3)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) at commit [772fa](https://github.com/NVIDIA/Megatron-LM/commit/772faca1f8d5030621b738cbd8e8bb2d8d28f6e6)
