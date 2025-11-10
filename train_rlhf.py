"""
        1. Load PEFT model and tokenizer.
        2. Load Reward model.
        3. Load Query dataset.
        4. Instantiate PPO Trainer.
        5. Loop-
            for each epoch 
                    get query
                    get response from peft model 
                    computer reward score from PEFT model
                    save the log 
                    run PPO

        6. Save model and tokenizer.

"""
import argparse
import sys
import torch
import pandas as pd
from copy import deepcopy
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GenerationConfig
)
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset, Dataset
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.trainer.ppo_trainer import PPOTrainer
import trl
print(trl.__version__)

# ----- Argument parser -----
parser = argparse.ArgumentParser(description="PPO fine-tuning with PEFT and reward model")
parser.add_argument("--dataset_path", type=str, default="/data/sample.json", help="Path to dataset json file")
parser.add_argument("--base_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model name or path")
parser.add_argument("--peft_model_path", type=str, default="/peft_model", help="Path to PEFT model")
parser.add_argument("--reward_model_path", type=str, default="/reward_model", help="Path to reward model")
parser.add_argument("--output_dir", type=str, default="ppo_finetuned_model", help="Directory to save final model and tokenizer")
args = parser.parse_args()

# ----- Configs (from args) -----
base_model_name = args.base_model_name
peft_model_path = args.peft_model_path
reward_model_path = args.reward_model_path
output_dir = args.output_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------Load dataset------
try:
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
except Exception as e:
    print(f"Failed to load dataset from {args.dataset_path}: {e}")
    sys.exit(1)

# ----- Tokenizers -----
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Failed to load tokenizer for base model {base_model_name}: {e}")
    sys.exit(1)

try:
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
except Exception as e:
    print(f"Failed to load reward tokenizer from {reward_model_path}: {e}")
    sys.exit(1)

# Load base model and PEFT
try:
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
except Exception as e:
    print(f"Failed to load base model {base_model_name}: {e}")
    sys.exit(1)

try:
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    model = AutoModelForCausalLMWithValueHead(peft_model)
    generation_config = GenerationConfig.from_pretrained(base_model_name)
    model.pretrained_model.generation_config = generation_config  # TRL expects this
    model = model.to(device)
except Exception as e:
    print(f"Failed to load or wrap PEFT model from {peft_model_path}: {e}")
    sys.exit(1)

# Reference model
try:
    ref_model = create_reference_model(model)
except Exception as e:
    print(f"Failed to create reference model: {e}")
    sys.exit(1)

# ----- Reward Model -----
try:
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path, num_labels=2).to(device)
except Exception as e:
    print(f"Failed to load reward model from {reward_model_path}: {e}")
    sys.exit(1)

# ----- Reward Function -----
def compute_reward(texts):
    """Return a list of positive-class scores for each text in texts."""
    try:
        tokens = reward_tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            reward_output = reward_model(**tokens)
        # return list of scores for positive class
        return reward_output.logits[:, 1].tolist()
    except Exception as e:
        print(f"Error computing reward: {e}")
        # return zeros so training can continue safely
        return [0.0 for _ in texts]

# ----- PPO Config -----
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
)

# ----- Dataset Wrapper -----
class QueryDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        return {"query": self._data[idx]["query"]}

    def __len__(self):
        return len(self._data)

#--------Pre-tokenization---------
def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample

try:
    dataset = dataset.map(tokenize, batched=False)
except Exception as e:
    print(f"Failed during dataset tokenization/map: {e}")
    sys.exit(1)

# ----- PPO Trainer -----
# ensure TRL expects these attributes
model.base_model_prefix = "model"
model.model = model.pretrained_model
model.generation_config = model.pretrained_model.generation_config

value_model = model

try:
    ppo_trainer = PPOTrainer(
        ppo_config,
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
    )
except Exception as e:
    print(f"Failed to initialize PPOTrainer: {e}")
    sys.exit(1)

# ----- Training Loop -----
log_path = "training_log.csv"
log_data = []

for i, batch in tqdm(enumerate(dataset)):
    try:
        # Tokenize input query string
        encoded = tokenizer(batch["query"],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512).to(device)

        # Generate response
        response_ids = model.generate(**encoded,
                                      max_new_tokens=100,
                                      do_sample=True,
                                      temperature=0.7,
                                      top_p=0.9,
                                      pad_token_id=tokenizer.pad_token_id)

        # Decode query and response
        query_text = batch["query"]
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Combine for reward
        full_convo = query_text + response_text
        reward_scores = compute_reward([full_convo])
        reward_score = reward_scores[0]

        reward_tensor = torch.tensor(reward_score).to(device)

        # PPO step
        ppo_trainer.step([query_text], [response_text], [reward_score])

        # Log
        log_data.append({"query": query_text, "response": response_text, "reward": float(reward_score)})
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}] Query: {query_text[:50]}... | Reward: {reward_score:.4f}")
            pd.DataFrame(log_data).to_csv(log_path, index=False)

    except Exception as e:
        print(f"Error in training loop at iteration {i}: {e}")
        # continue to next batch
        continue

# ----- Save Final Model -----

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
pd.DataFrame(log_data).to_csv(log_path, index=False)
print("Training complete and model saved.")
