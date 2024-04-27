from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def Inference_func(Prompt, question, model):
    try:
        response = client.chat.completions.create(
                model= model,
                temperature=0,
                messages=[
                    {"role": "system", "content": Prompt},
                    {"role": "user", "content": question},
                    ]
                )
        output = response.choices[0].message.content

        return output
    
    except Exception as e:
        print(e)