import os
import argparse
from datetime import datetime
import torch
import json
from llmtuner import run_exp
import wandb

# Determine the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
config_file_name = "finetuning_configs.json"
config_path = os.path.join(script_dir, config_file_name)
default_output_folder = os.path.join(script_dir, "../../../checkpoints")

# Setup argument parser
parser = argparse.ArgumentParser(description="Fine-tune a model with specified parameters")
parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of the model")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
parser.add_argument("--config_name", type=str, default="default-config", help="Name of the configuration to use")
parser.add_argument("--output_folder", type=str, default=default_output_folder, help="The output directory for the trained checkpoint")

args = parser.parse_args()

# GPU availability check
try:
    assert torch.cuda.is_available() is True
except AssertionError:
    print("Please set up a GPU before using LLaMA Factory")
    exit()

# Generate a unique output directory based on model, dataset, config name, and current time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name_short = args.model_name_or_path.split("/")[-1]
run_name = f"{model_name_short}_{args.dataset}_{args.config_name}_{current_time}"
output_dir = os.path.join(args.output_folder, run_name)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Log in to Weights & Biases
wandb.login()

# Configure Weights & Biases project
wandb_project = "BA_Text-To-SQL"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
wandb.init(project=wandb_project, name=run_name)

# Load configurations from the specified JSON file
with open(config_path, 'r') as config_file:
    configs = json.load(config_file)

# Select the specific configuration by its name
if args.config_name in configs:
    exp_config = configs[args.config_name]
else:
    print(f"Configuration '{args.config_name}' not found. Exiting.")
    exit()

# Update the selected configuration with dynamic values from command-line arguments
exp_config.update({
    "model_name_or_path": args.model_name_or_path,
    "dataset": args.dataset,
    "output_dir": output_dir,
})

# Run experiment with the selected and updated configuration
run_exp(exp_config)
