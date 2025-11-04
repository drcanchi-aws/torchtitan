## Neuron Testing

#### Make runs deterministic (validated with TP=1/single core exec)
1. Add to toml:

```bash
[checkpoint]
enable = true
initial_load_path = "outputs/checkpoint-init-qwen3_8b_reduce_size_tp4/step-0"
initial_load_model_only = true
folder = "checkpoint-train-qwen3_8b_reduce_size_tp4-gbs1-autocast"
interval = 200
last_save_model_only = true
export_dtype = "float32"
async_mode = "disabled" # ["disabled", "async", "async_with_pinned_mem"]
```

2. Save step0/seed ckpt: https://github.com/drcanchi-aws/torchtitan/blob/33ec0ce4475e6c1cadee08cfdb5fb814b63091c7/docs/checkpoint.md?plain=1#L55

```bash
./run_neuron_train.sh --checkpoint.enable --checkpoint.create_seed_checkpoint
```