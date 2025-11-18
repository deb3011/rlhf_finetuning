# RLHF PPO Finetuning (PEFT + TRL)

This repository contains a script to perform reinforcement learning fine-tuning (RLHF) of a PEFT-finetuned causal LLM using Proximal Policy Optimization (PPO) via the TRL library.
TRL version - 0.11.4

Files/folders (expected layout)

- `train_rlhf.py` - main training script.
- `requirements.txt` - Python dependencies used by the project.
- `data/` - directory containing a JSON dataset file. Each JSON object should contain a `query` field (e.g. `{ "query": "<prompt text>" }`).
- `peft_model/` - directory containing your PEFT-finetuned adapter/checkpoint (the script loads the PEFT model from this folder).
- `reward_model/` - directory containing a sequence classification model used as the reward model (script expects a binary classifier: logits -> [neg, pos]).
- `training_log.csv` (generated) - CSV log of queries, responses and reward scores.
- `ppo_finetuned_model/` (generated) - output directory where final model and tokenizer are saved (configurable via `--output_dir`).

Quick overview of `train_rlhf.py`

1. Argument parsing
   - `--dataset_path` : path to dataset JSON (default `data/sample.json` in the script). The script uses `datasets.load_dataset("json")` and expects a `train` split.
   - `--base_model_name` : base LM name or path (HF model id or local path).
   - `--peft_model_path` : path to PEFT model directory (e.g. `peft_model/`).
   - `--reward_model_path` : path to reward model directory (e.g. `reward_model/`).
   - `--output_dir` : directory where the final model/tokenizer will be saved.

2. Main steps performed
   - Load the dataset using `datasets.load_dataset`.
   - Load the base tokenizer (and set `pad_token` if missing).
   - Load the reward tokenizer and reward classification model (moved to device).
   - Load the base causal model, attach PEFT weights via `peft.PeftModel`, wrap with TRL value head (`AutoModelForCausalLMWithValueHead`) and create a reference model.
   - Instantiate `PPOTrainer` (TRL) with the model, reference model and reward model.
   - Iterate over the dataset, generate responses from the PEFT model, compute reward scores using the reward model, call `ppo_trainer.step()` and log results.
   - Save the final model and tokenizer to `--output_dir` and persist `training_log.csv`.

Notes about dataset and reward model

- dataset JSON: each record must contain a `query` string. Example file content:
  [
    {"query": "Explain RLHF in simple terms."},
    {"query": "Write a short poem about autumn."}
  ]

- reward model: script expects a sequence-classification model with `num_labels=2` where logits[:, 1] corresponds to the positive reward score. If your reward model outputs different label ordering or uses regression, adapt `compute_reward` accordingly.

Runtime & hardware

- The script will use CUDA if available; otherwise CPU. Large models (Mistral-7B, etc.) require a GPU with enough memory. The script loads the base model with `torch_dtype=torch.bfloat16` (adjust if your hardware or model doesn't support bfloat16).

Safe/robust behavior

- The script wraps critical steps in try/except blocks (dataset loading, tokenizer/model loading, tokenization mapping, PPOTrainer creation, training loop and final save) and exits with messages on failure.

How to run

1. Place your files in the expected folders:
   - put the dataset JSON in `data/` (or pass a path to `--dataset_path`)
   - put the PEFT checkpoint under `peft_model/`
   - put the reward model under `reward_model/`

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Example run (adjust flags as needed):

```bash
python train_rlhf.py \
  --dataset_path data/sample.json \
  --base_model_name <your-base-model-or-hf-id> \
  --peft_model_path peft_model \
  --reward_model_path reward_model \
  --output_dir ppo_finetuned_model
```

Tips and troubleshooting

- If tokenizer has no `pad_token`, the script sets `pad_token = eos_token`. For some models you might need to set `tokenizer.pad_token_id` explicitly.
- If model download fails, check internet connection or use local cached model paths.
- Large model memory issues: consider model offloading, quantization, or using smaller models for experimentation.
- If reward scores appear meaningless, validate the reward model independently by calling it on sample concatenated prompts+responses and inspect logits.
- Consider switching to a smaller `batch_size` or `max_new_tokens` to reduce memory usage.

Important links

- PPO (Proximal Policy Optimization) paper — Schulman et al., 2017: https://arxiv.org/abs/1707.06347
- TRL (TRL library by Hugging Face): https://github.com/huggingface/trl
- PEFT (Parameter-Efficient Fine-Tuning): https://github.com/huggingface/peft
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- Hugging Face Datasets documentation: https://huggingface.co/docs/datasets

