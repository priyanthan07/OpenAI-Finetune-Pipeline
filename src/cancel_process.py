import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def cancel_process(finetune_id):
    
    client = OpenAI(
        api_key=OPENAI_API_KEY
        
    )
    print("inside cancel")
    # Cancel a job
    status = client.fine_tuning.jobs.cancel(finetune_id)
    
    return status