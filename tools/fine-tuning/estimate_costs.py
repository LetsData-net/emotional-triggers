import json
import tiktoken
import numpy as np

"""
based on open-ai cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/Chat_finetuning_data_prep.ipynb
"""

MAX_TOKENS_PER_EXAMPLE = 10e6
TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 50
MAX_TARGET_EXAMPLES = 200
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

DATA_DIR = 'data/ft'

encoding = tiktoken.get_encoding("cl100k_base")


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


def cost_estimation(dataset, n_epochs: int = 5, price: int | None = None):
    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > MAX_TOKENS_PER_EXAMPLE for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the {MAX_TOKENS_PER_EXAMPLE} token limit, "
          f"they will be truncated during fine-tuning")

    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")

    if price is not None:
        charged_price = n_billing_tokens_in_dataset * price / 1000000
        print(f"By default, you'll be charged for approx ${np.round(charged_price, 2)}")


def main():

    for persona_name in ['persona_1', 'persona_2']:
        data_path = f"{DATA_DIR}/{persona_name}_ft_training.jsonl"

        with open(data_path, 'r') as f:
            dataset = [json.loads(line) for line in f]

        cost_estimation(dataset=dataset, n_epochs=TARGET_EPOCHS, price=8)


if __name__ == '__main__':
    main()
