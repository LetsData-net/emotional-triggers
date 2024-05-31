import json
import os

from tqdm import tqdm

from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from inference.zero_shot_inference import load_dataset_for_completion, load_personas_context

MODEL_NAME = "gpt-3.5-turbo-1106"
PROMPT_PATH = os.environ['PROMPT_PATH']
TEMPERATURE = 0.2
MAX_TOKENS = 4096
EXAMPLES_NUM = 10
RANDOM_SEEDS = [7, 33, 42]
APPROACH_NAME = f"context_few_shot_{EXAMPLES_NUM}"
DATA_DIR = 'data'

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = f'context_based_triggers : {APPROACH_NAME}'


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


def load_backup(approach_name: str = APPROACH_NAME):
    annotations = dict()
    for persona_name in ["persona_1", "persona_2"]:
        annotations[persona_name] = dict()
        for seed in RANDOM_SEEDS:
            if os.path.exists(f"{DATA_DIR}/backup/{persona_name}_{approach_name}_seed={seed}.jsonl"):
                annotations[persona_name][seed] = dict()
                with open(f"{DATA_DIR}/backup/{persona_name}_{approach_name}_seed={seed}.jsonl") as lines:
                    for line in lines:
                        line_results = json.loads(line)
                        annotations[persona_name][seed][line_results['id']] = line_results
    return annotations


def get_prompt(persona_name):

    template_prompt = hub.pull(PROMPT_PATH)

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
    return prompt


def main():

    publications_subset = load_dataset_for_completion()
    personas_dict = load_personas_context()

    os.makedirs(f'{DATA_DIR}/backup', exist_ok=True)
    os.makedirs(f'{DATA_DIR}/completed', exist_ok=True)

    annotations = load_backup()

    for persona_name in ["persona_1", "persona_2"]:

        prompt = get_prompt(persona_name)

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
