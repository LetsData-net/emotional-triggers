import json
import os

from tqdm import tqdm

from langchain.chat_models import ChatOpenAI

from inference.zero_shot_inference import get_prompt, load_dataset_for_completion, load_personas_context
from inference.few_shot_inference import load_backup

OPENAI_ORG = os.environ['OPENAI_ORG']
MODEL_NAMES_DICT = {
    "persona_1": f"ft:gpt-3.5-turbo-1106:{OPENAI_ORG}:persona-1:9MYWd3x9:ckpt-step-150",
    "persona_2": f"ft:gpt-3.5-turbo-1106:{OPENAI_ORG}:persona-2:9MYaUyVH:ckpt-step-150",
}
PROMPT_PATH = os.environ['PROMPT_PATH']
TEMPERATURE = 0.2
MAX_TOKENS = 4096
RANDOM_SEEDS = [7, 33, 42]
APPROACH_NAME = "context_fine_tuning"
DATA_DIR = 'data'

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = f'context_based_triggers : {APPROACH_NAME}'


def main():

    prompt = get_prompt()

    publications_subset = load_dataset_for_completion()

    personas_dict = load_personas_context()

    os.makedirs(f'{DATA_DIR}/backup', exist_ok=True)
    os.makedirs(f'{DATA_DIR}/completed', exist_ok=True)

    annotations = load_backup(approach_name=APPROACH_NAME)

    for persona_name in ["persona_1", "persona_2"]:

        model_name = MODEL_NAMES_DICT[persona_name]
        print(f"model - {model_name}")

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
                    "model_name": model_name,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                }

                llm_chat = ChatOpenAI(temperature=TEMPERATURE, model_name=model_name, max_tokens=MAX_TOKENS,
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
