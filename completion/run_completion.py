import json
import os

import pandas as pd

from tqdm import tqdm

from langchain import hub
from langchain.chat_models import ChatOpenAI


MODEL_NAME = "gpt-3.5-turbo-1106"
PROMPT_PATH = os.environ['PROMPT_PATH']
TEMPERATURE = 0.3
MAX_TOKENS = 4096
APPROACH_NAME = "context_pure"
DATA_DIR = 'data'


def main():
    personas_dict = json.load(open(f"{DATA_DIR}/personas_dict.json"))

    publications_subset = pd.read_csv(f"{DATA_DIR}/labelbox_sample.csv")

    evaluation_ids = json.load(open(f"{DATA_DIR}/data_split_ids.json"))["evaluation"]

    publications_subset = publications_subset[publications_subset.id.isin(evaluation_ids)].reset_index(drop=True)

    print(f"Will run chat completion for {len(publications_subset)} publications.")

    prompt = hub.pull(PROMPT_PATH)

    model_kwargs = {
        "response_format": {"type": "json_object"}
    }

    llm_chat = ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME, max_tokens=MAX_TOKENS,
                          model_kwargs=model_kwargs)

    annotations = dict()

    for persona_name in ["assistant", "persona_1", "persona_2"]:

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
            }

            annotations[persona_name][row_id] = json_string

            approach_name = "no_context" if persona_name != "assistant" else APPROACH_NAME
            with open(f"{DATA_DIR}/backup/{persona_name}_{approach_name}.jsonl", "a") as out_file:
                out_file.write(json.dumps(json_string, ensure_ascii=False) + "\n")

    for persona_name, persona_annotations in annotations.items():
        approach_name = "no_context" if persona_name == "assistant" else APPROACH_NAME
        with open(f"{DATA_DIR}/completed/{persona_name}_{approach_name}.jsonl", "a") as out_file:
            for annotation in persona_annotations.values():
                out_file.write(json.dumps(annotation, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
