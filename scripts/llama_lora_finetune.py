import argparse
import torch
from llmtuner import run_exp
import wandb, os

# Setup argument parser
parser = argparse.ArgumentParser(description="Fine-tune a model with specified parameters")
parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of the model")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
parser.add_argument("--output_folder", type=str, required=True, help="Folder name for the output directory")

# Add any other parameters you might want to configure via command-line arguments
parser.add_argument("--learning_rate", type=float, default=5e-05, help="Learning rate")
parser.add_argument("--num_train_epochs", type=float, default=5.0, help="Number of training epochs")
parser.add_argument("--max_samples", type=int, default=500, help="Maximum number of samples")

args = parser.parse_args()

# GPU availability check
try:
  assert torch.cuda.is_available() is True
except AssertionError:
  print("Please set up a GPU before using LLaMA Factory")
  exit()

# Log in to Weights & Biases
wandb.login()

# Configure Weights & Biases project
wandb_project = "codellama-7b-sql-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

# Prepend the path to the output_folder
output_dir = f"saves/Custom/lora/{args.output_folder}"

# Run experiment with specified arguments
run_exp(dict(
  stage="sft",
  do_train=True,
  model_name_or_path=args.model_name_or_path,
  dataset=args.dataset,
  template="default",
  finetuning_type="lora",
  lora_target="all",
  output_dir=output_dir,
  per_device_train_batch_size=4,
  gradient_accumulation_steps=4,
  lr_scheduler_type="cosine",
  logging_steps=10,
  save_steps=100,
  learning_rate=args.learning_rate,
  num_train_epochs=args.num_train_epochs,
  max_samples=args.max_samples,
  max_grad_norm=1.0,
  fp16=True,
  report_to="wandb"
))
