from openai import OpenAI


if __name__ == '__main__':

    client = OpenAI()

    model_name = "gpt-3.5-turbo"

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
            },
            {
                "role": "user",
                "content": "Compose a poem that explains the concept of recursion in programming."
            }
        ]
    )

    print(completion.choices[0].message)
