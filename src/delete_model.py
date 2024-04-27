import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def delete_model(finetuned_model_name):
    client = OpenAI(
        api_key=OPENAI_API_KEY
    )
    
    # Delete a fine-tuned model 
    status = client.models.delete(finetuned_model_name)
    
    return status