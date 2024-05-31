import json
import os

import pandas as pd

from tqdm import tqdm

from langchain import hub
from langchain.chat_models import ChatOpenAI

MODEL_NAME = "gpt-3.5-turbo-1106"
PROMPT_PATH = os.environ['PROMPT_PATH']
TEMPERATURE = 0.2
MAX_TOKENS = 4096
RANDOM_SEEDS = [7, 33, 42]
APPROACH_NAME = "context_zero_shot"
DATA_DIR = 'data'

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = f'context_based_triggers : {APPROACH_NAME}'


def load_dataset_for_completion(publications_file_name: str = 'labelbox_sample.csv',
                                data_split_file_name: str = 'data_split_ids.json') -> pd.DataFrame:

    publications_subset = pd.read_csv(f"{DATA_DIR}/{publications_file_name}")

    evaluation_ids = json.load(open(f"{DATA_DIR}/{data_split_file_name}"))["evaluation"]

    publications_subset = publications_subset[publications_subset.id.isin(evaluation_ids)].reset_index(drop=True)

    print(f"Will run chat completion for {len(publications_subset)} publications.")

    return publications_subset


def load_personas_context(personas_context_file_name: str = 'personas_dict.json') -> dict:
    return json.load(open(f"{DATA_DIR}/{personas_context_file_name}"))


def get_prompt():
    prompt = hub.pull(PROMPT_PATH)
    return prompt


def load_backup():
    annotations = dict()
    for persona_name in ["persona_1", "persona_2", "assistant"]:
        approach_name = "no_context" if persona_name == "assistant" else APPROACH_NAME
        if os.path.exists(f"{DATA_DIR}/backup/{persona_name}_{approach_name}.jsonl"):
            annotations[persona_name] = dict()
            with open(f"{DATA_DIR}/backup/{persona_name}_{approach_name}.jsonl") as lines:
                for line in lines:
                    line_results = json.loads(line)
                    annotations[persona_name][line_results[0]['id']] = line_results
    return annotations


def main():

    prompt = get_prompt()

    publications_subset = load_dataset_for_completion()

    personas_dict = load_personas_context()

    os.makedirs(f'{DATA_DIR}/backup', exist_ok=True)
    os.makedirs(f'{DATA_DIR}/completed', exist_ok=True)

    annotations = load_backup()

    for persona_name in ["persona_1", "persona_2", "assistant"]:

        persona_context = personas_dict[persona_name]
        print(f"persona - {persona_context}")

        if persona_name not in annotations:
            annotations[persona_name] = {}

        for seed in RANDOM_SEEDS:

            if seed not in annotations[persona_name]:
                annotations[persona_name][seed] = {}

            for i, row in tqdm(publications_subset.iterrows(), total=len(publications_subset)):

                row_id = row["id"]

                if row_id in annotations[persona_name][seed]:
                    print(f"skipping {row_id} as already in annotations")
                    continue

                publication = row["text"]

                output_json = {
                    "id": row_id,
                    "model_name": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                }

                llm_chat = ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME, max_tokens=MAX_TOKENS,
                                      model_kwargs={"response_format": {"type": "json_object"}, "seed": seed},
                                      request_timeout=10)
                try:
                    response = llm_chat.invoke(prompt.format(**{"CONTEXT": persona_context,
                                                                "PUBLICATION": publication}), )
                    json_string = json.loads(response.content)
                except Exception as ex:
                    print(ex)
                    json_string = {"response": "failed to generate"}

                print(json_string)

                output_json['seed'] = seed
                output_json['output'] = json_string

                annotations[persona_name][seed][row_id] = output_json

                with open(f"{DATA_DIR}/backup/{persona_name}_{APPROACH_NAME}_seed={seed}.jsonl", "a") as out_file:
                    out_file.write(json.dumps(output_json, ensure_ascii=False) + "\n")

    for persona_name, persona_annotations in annotations.items():
        for seed, seed_results in persona_annotations.items():
            with open(f"{DATA_DIR}/completed/{persona_name}_{APPROACH_NAME}_seed={seed}.jsonl", "w") as out_file:
                for annotation in seed_results.values():
                    out_file.write(json.dumps(annotation, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
