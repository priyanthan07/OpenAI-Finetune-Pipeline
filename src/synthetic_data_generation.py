import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

FILE_PATH = "./tmp"


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH)
     
    
def data_generation(context, prompt, count, assistant, user):      


    data_format = f"""Generate the most critical conversations that remain within our context.
        
        Output the response as a JSON object in the following format.
        {{
            "conversation" : [
                {{
                    "{user}": <response of the {user}>,
                    "{assistant}": <response of the {assistant}>                    
                }},
                <more objects>
            ]
        }}

    """

    PROMPT = prompt + data_format # complete prompt for the data generation
    

    data = []
    try:
        for i in range(count):

            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                temperature=0.9,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": PROMPT},
                    ]
                )
            output_conv = response.choices[0].message.content

            # Assuming the conversation is in the desired format
            conversation_data = json.loads(output_conv)
            conversations = conversation_data['conversation']

            if len(conversations) > 0:
                for conversation_item in conversations:
                    if assistant in conversation_item and user in conversation_item:
                        data.append(conversation_item)

        full_path = os.path.join(FILE_PATH, "syntheic_data.json")

        with open(full_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    except Exception as e:
        print(e)

        