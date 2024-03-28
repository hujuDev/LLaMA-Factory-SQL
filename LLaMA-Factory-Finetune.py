import torch
from llmtuner import run_exp
import wandb, os

try:
  assert torch.cuda.is_available() is True
except AssertionError:
  print("Please set up a GPU before using LLaMA Factory: https://medium.com/mlearning-ai/training-yolov4-on-google-colab-316f8fff99c6")


wandb.login()

wandb_project = "codellama-7b-sql-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

run_exp(dict(
  stage="sft",
  do_train=True,
  model_name_or_path="codellama/CodeLlama-7b-hf",
  dataset="spider_sql_custom",
  template="default",
  finetuning_type="lora",
  lora_target="all",
  output_dir="saves/Custom/lora/CodeLlama-7b-hf_run#3",
  per_device_train_batch_size=4,
  gradient_accumulation_steps=4,
  lr_scheduler_type="cosine",
  logging_steps=10,
  save_steps=100,
  learning_rate=5e-05,
  num_train_epochs=5.0,
  max_samples=500,
  max_grad_norm=1.0,
  fp16=True,
  report_to="wandb"
))