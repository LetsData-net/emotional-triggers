import json
import os

import pandas as pd

from tqdm import tqdm

from langchain import hub
from langchain.chat_models import ChatOpenAI


OPENAI_ORG = os.environ['OPENAI_ORG']
MODEL_NAMES_DICT = {
    "persona_1": f"ft:gpt-3.5-turbo-1106:{OPENAI_ORG}:persona-1:9MYWd3x9:ckpt-step-150",
    "persona_2": f"ft:gpt-3.5-turbo-1106:{OPENAI_ORG}:persona-2:9MYaUyVH:ckpt-step-150",
}
PROMPT_PATH = os.environ['PROMPT_PATH']
TEMPERATURE = 0.3
MAX_TOKENS = 4096
APPROACH_NAME = "context_fine_tuning"
DATA_DIR = 'data'


def main():
    personas_dict = json.load(open(f"{DATA_DIR}/personas_dict.json"))

    publications_subset = pd.read_csv(f"{DATA_DIR}/labelbox_sample.csv")

    evaluation_ids = json.load(open(f"{DATA_DIR}/data_split_ids.json"))["evaluation"]

    publications_subset = publications_subset[publications_subset.id.isin(evaluation_ids)].reset_index(drop=True)

    prompt = hub.pull(PROMPT_PATH)

    model_kwargs = {
        "response_format": {"type": "json_object"}
    }

    annotations = dict()

    for persona_name in ["persona_1", "persona_2"]:

        model_name = MODEL_NAMES_DICT[persona_name]
        print(f"model - {model_name}")

        llm_chat = ChatOpenAI(temperature=TEMPERATURE, model_name=model_name, max_tokens=MAX_TOKENS,
                              model_kwargs=model_kwargs)

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
                "model_name": model_name,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
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
