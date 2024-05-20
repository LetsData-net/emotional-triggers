import os

from openai import OpenAI

WANDB_PROJECT_NAME = os.environ['WANDB_PROJECT_NAME']  # wandb project name
WANDB_ENTITY = os.environ["WANDB_ENTITY"]  # wandb username
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

MODEL_NAME = "gpt-3.5-turbo-1106"
N_EPOCHS = 4
RANDOM_SEED = 42
DATA_DIR = 'data/ft'


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    for persona_name in ["persona_1", "persona_2"]:

        print(f'Setting ft for {persona_name}')

        training_file = client.files.create(
            file=open(f"{DATA_DIR}/{persona_name}_ft_training.jsonl", "rb"),
            purpose="fine-tune"
        )
        print(training_file.id)

        validation_file = client.files.create(
            file=open(f"{DATA_DIR}/{persona_name}_ft_validation.jsonl", "rb"),
            purpose="fine-tune"
        )
        print(validation_file.id)

        job = client.fine_tuning.jobs.create(
            training_file=training_file.id,
            validation_file=validation_file.id,
            model=MODEL_NAME,
            hyperparameters={
                "n_epochs": N_EPOCHS,
            },
            suffix=persona_name,
            seed=RANDOM_SEED,
            integrations=[{
                "type": "wandb",
                "wandb": {
                    "project": WANDB_PROJECT_NAME,
                    "entity": WANDB_ENTITY,
                    "tags": [persona_name]
                }
            }]
        )

        print(job.id)


if __name__ == '__main__':
    main()
