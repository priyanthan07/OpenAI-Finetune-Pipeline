import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def finetune_func( train_id, test_id):
    global fine_tune_id
    
    client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    response = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=test_id,
        model='gpt-3.5-turbo-1106' 
    )

    fine_tune_id = response.id
    
    return fine_tune_id