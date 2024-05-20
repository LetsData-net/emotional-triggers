import json

from copy import deepcopy

import numpy as np
import pandas as pd

"""
Guidelines: https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
Example format:
{  
    "messages": [
        {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}
    ]
}
"""

DATA_DIR = 'data'
PROMPTS_DIR = 'prompts'
annotator2persona_mapping = {"persona_1": "d16ec044", "persona_2": "3fefb024"}


def main():
    personas_dict = json.load(open(f"{DATA_DIR}/personas_dict.json"))
    prompt = json.load(open(f"{PROMPTS_DIR}/evaluate_publication_by_persona.json"))

    ft_examples = dict()

    for persona_name, persona_context in personas_dict.items():

        ft_examples[persona_name] = dict()

        annotator_hash = annotator2persona_mapping.get(persona_name)

        if annotator_hash is None:
            continue

        for split_name in ["ft_training", "ft_validation"]:

            split_annotations = pd.read_csv(f"{DATA_DIR}/train_test_eval_split/{annotator_hash}_{split_name}.csv")

            ft_examples[persona_name][split_name] = []

            for annotation in split_annotations.to_dict("records"):

                ft_example = deepcopy(prompt)
                ft_example[0]["content"] = ft_example[0]["content"].replace("{CONTEXT}", personas_dict[persona_name])
                ft_example[1]["content"] = ft_example[1]["content"].replace("```{PUBLICATION}```", annotation["text"])

                for col in ["context_locations", "created_at", "created_by", "text", "id"]:
                    annotation.pop(col)

                for col in ["emotion", "emotion_class", "trigger_level", "context", "description"]:
                    assert col in annotation, f"missing column: {col} in annotation: {annotation}"

                if annotation["description"] in [None, np.nan]:
                    raise ValueError(annotation)

                ft_example.append({
                    "role": "assistant",
                    "content": json.dumps(annotation, ensure_ascii=False)
                })

                ft_examples[persona_name][split_name].append({"messages": ft_example})

    for persona_name, splits_data in ft_examples.items():
        for split_name, split_examples in splits_data.items():
            with open(f"{DATA_DIR}/ft/{persona_name}_{split_name}.jsonl", "w") as output_file:
                for example in split_examples:
                    output_file.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()
