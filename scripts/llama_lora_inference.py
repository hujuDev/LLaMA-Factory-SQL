import argparse
from llmtuner import ChatModel

# Setup argument parser
parser = argparse.ArgumentParser(description="Run Chat Model with specified model and adapter")
parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of the model")
parser.add_argument("--adapter_name_or_path", type=str, required=True, help="Path or name of the adapter")
args = parser.parse_args()

# Initialize ChatModel with command-line arguments
chat_model = ChatModel(dict(
    model_name_or_path=args.model_name_or_path,
    adapter_name_or_path=args.adapter_name_or_path,
    finetuning_type="lora",
    template="default",
))

messages = []

while True:
    query = input("\nUser: ")
    if query.strip().lower() == "exit":
        break
    if query.strip().lower() == "clear":
        messages = []
        continue

    messages.append({"role": "user", "content": query})
    print("Assistant: ", end="", flush=True)
    response = ""
    for new_text in chat_model.stream_chat(messages):
        print(new_text, end="", flush=True)
        response += new_text
    print()
    messages.append({"role": "assistant", "content": response})
