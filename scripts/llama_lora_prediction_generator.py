import argparse
import json
from llmtuner import ChatModel
from tqdm import tqdm  # Import tqdm for the progress bar

# Setup argument parser
parser = argparse.ArgumentParser(description="Generate responses for a JSON input file using ChatModel")
parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of the model")
parser.add_argument("--adapter_name_or_path", type=str, required=True, help="Path or name of the adapter")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
parser.add_argument("--num_entries", type=int, help="Optional: Number of entries to process (if not set, all entries will be processed)")
args = parser.parse_args()

# Initialize ChatModel with command-line arguments
chat_model = ChatModel(dict(
    model_name_or_path=args.model_name_or_path,
    adapter_name_or_path=args.adapter_name_or_path,
    finetuning_type="lora",
    template="default",
))

# Load the JSON input file
with open(args.input_file, "r") as file:
    test_dataset = json.load(file)

# If --num_entries is provided, limit the dataset to that number of entries
if args.num_entries is not None:
    test_dataset = test_dataset[:args.num_entries]

# Initialize a list to collect the outputs
outputs = []

# Process each entry in the test dataset
for entry in tqdm(test_dataset, desc="Generating predictions"):
    input = entry["Input"]
    # Generate response for the question
    messages = [{"role": "user", "content": input}]
    response = ""
    for new_text in chat_model.stream_chat(messages):
        response += new_text
    # Append the response to outputs
    outputs.append(response)

# Write the outputs to predicted.txt
with open("predicted.txt", "w") as file:
    for output in outputs:
        file.write(output + "\n")
