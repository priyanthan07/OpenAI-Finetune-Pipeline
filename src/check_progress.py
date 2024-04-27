import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_job(fine_tuning_job):
    info = {
        "Fine Tuning Job ID": fine_tuning_job.id,
        "Created At": fine_tuning_job.created_at,
        "Error": fine_tuning_job.error,
        "Fine Tuned Model": fine_tuning_job.fine_tuned_model,
        "Finished At": fine_tuning_job.finished_at,
        "Hyperparameters: Epochs": fine_tuning_job.hyperparameters.n_epochs,
        "Hyperparameters: Batch Size": fine_tuning_job.hyperparameters.batch_size,
        "Hyperparameters: Learning Rate Multiplier": fine_tuning_job.hyperparameters.learning_rate_multiplier,
        "Model": fine_tuning_job.model,
        "Object Type": fine_tuning_job.object,
        "Organization ID": fine_tuning_job.organization_id,
        "Result Files": fine_tuning_job.result_files,
        "Status": fine_tuning_job.status,
        "Trained Tokens": fine_tuning_job.trained_tokens,
        "Training File": fine_tuning_job.training_file,
        "Validation File": fine_tuning_job.validation_file
    }

    label_width = max(len(label) for label in info) + 2  # Adding 2 for extra spacing

    for label, value in info.items():
        print(f"{label.ljust(label_width)}: {value}")
    return info



def progress(finetune_id):
    client = OpenAI()        #api_key=OPENAI_API_KEY
        
    # Retrieve the state of a fine-tune
    fine_tuning_job = client.fine_tuning.jobs.retrieve(finetune_id)
    
    info = get_job(fine_tuning_job)
        
    return  info  
