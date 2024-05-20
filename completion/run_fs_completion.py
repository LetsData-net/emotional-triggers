import json
import os

import pandas as pd

from tqdm import tqdm

from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

MODEL_NAME = "gpt-3.5-turbo-1106"
PROMPT_PATH = os.environ['PROMPT_PATH']
TEMPERATURE = 0.3
MAX_TOKENS = 4096
EXAMPLES_NUM = 10
APPROACH_NAME = f"context_few_shot_{EXAMPLES_NUM}"
DATA_DIR = 'data'


def load_examples(persona_name):
    examples = []
    with open(f"{DATA_DIR}/fs/{persona_name}_fs_training.jsonl") as in_file:
        for line in in_file:
            parsed_line = json.loads(line)
            examples.append({
                "input": parsed_line["messages"][0]["content"],
                "output": parsed_line["messages"][1]["content"],
            })

    return examples[:EXAMPLES_NUM]


def main():
    personas_dict = json.load(open(f"{DATA_DIR}/personas_dict.json"))

    publications_subset = pd.read_csv(f"{DATA_DIR}/labelbox_sample.csv")

    evaluation_ids = json.load(open(f"{DATA_DIR}/data_split_ids.json"))["evaluation"]

    publications_subset = publications_subset[publications_subset.id.isin(evaluation_ids)].reset_index(drop=True)

    model_name = MODEL_NAME
    model_kwargs = {
        "response_format": {"type": "json_object"}
    }

    template_prompt = hub.pull(PROMPT_PATH)

    llm_chat = ChatOpenAI(temperature=TEMPERATURE, model_name=model_name, max_tokens=MAX_TOKENS,
                          model_kwargs=model_kwargs)

    annotations = dict()

    for persona_name in ["persona_1", "persona_2"]:

        examples = load_examples(persona_name)

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "{input}"),
                ("assistant", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        print(few_shot_prompt.format())

        prompt = ChatPromptTemplate.from_messages(
            [
                template_prompt.messages[0],
                few_shot_prompt,
                template_prompt.messages[1],
            ]
        )

        persona_context = personas_dict[persona_name]
        print(f"persona - {persona_context}")

        if persona_name not in annotations:
            annotations[persona_name] = {}

        for i, row in tqdm(publications_subset.iterrows(), total=len(publications_subset)):
            row_id = row["id"]

            if row_id in annotations[persona_name]:
                print(f"skipping {row_id} as already in annotations")
                continue

            publication = row["text"]

            try:
                print(prompt.format(**{"CONTEXT": persona_context, "PUBLICATION": publication}))
            except Exception as ex:
                print(ex)

            try:
                response = llm_chat.invoke(prompt.format(**{"CONTEXT": persona_context, "PUBLICATION": publication}))
                json_string = json.loads(response.content)
            except Exception as ex:
                print(ex)
                json_string = {"response": "failed to generate"}

            print(json_string)

            json_string["id"] = row_id
            json_string["model_params"] = {
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "examples_num": APPROACH_NAME,
            }

            annotations[persona_name][row_id] = json_string

            with open(f"{DATA_DIR}/backup/{persona_name}_{APPROACH_NAME}.jsonl", "a") as out_file:
                out_file.write(json.dumps(json_string, ensure_ascii=False) + "\n")

    for persona_name, persona_annotations in annotations.items():
        with open(f"{DATA_DIR}/completed/{persona_name}_{APPROACH_NAME}.jsonl", "w") as out_file:
            for annotation in persona_annotations.values():
                out_file.write(json.dumps(annotation, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
