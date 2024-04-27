from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
client = OpenAI()

def get_total_finetuned_model_count():
    
    Jobs = client.fine_tuning.jobs.list()
    model_details = []

    for job in Jobs:
        
        if job.fine_tuned_model != None:
            detail = {
                "Id " : job.id,
                "modelName " : job.fine_tuned_model
            }
            model_details.append(detail)
    DF = pd.DataFrame(model_details)

    return DF



    