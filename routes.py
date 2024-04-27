from fastapi import FastAPI, File, UploadFile, Response, APIRouter
from fastapi.responses import FileResponse, JSONResponse

from src.synthetic_data_generation import data_generation
from src.upload_data import upload
from src.finetune_process import finetune_func
from src.check_progress import progress
from src.cancel_process import cancel_process
from src.delete_model import delete_process
from src.data_preprocess import preprocess
from src.finetuned_models import get_total_finetuned_model_count

from dto.fine_tune_dto import FineTune

router = APIRouter()

# generate synthetic data
@router.post("/generate-data")
async def generate_synthetic_data(context : str, prompt: str, count : int,  AI : str, USER : str):
    synthetic_data = data_generation(context, prompt, count, AI, USER)
    return synthetic_data

# finetune the model 
@router.post("/fine-tune")
async def fine_tune(model: str, body_params : FineTune, context : str, ai_role : str, user_role : str, job_id: str): 
    # body_params_dict = body_params.dict
    # print(finetune_data)
    train_jsonl, test_jsonl = preprocess(body_params.finetune_data, context, ai_role, user_role)                                    
    train_id, test_id = await upload(job_id,train_jsonl, test_jsonl)
    finetune_id = await finetune_func(model,train_id, test_id)
    return finetune_id

# retrieving the progress of finetune
@router.post("/get-progress")
async def get_progress(finetune_id: str):
    
    status = await progress(finetune_id)
    return status

# cancel the finetune
@router.post("/cancel-finetune")
async def cancel(finetune_id: str):
    print(finetune_id)
    response = await cancel_process(finetune_id)
    return response

@router.post("/Total-finetuned-models")
async def fine_tuned_models():  
    models = get_total_finetuned_model_count()                                          
    return models

# delete the model
@router.post("/delete-model")
async def delete(finetuned_model_name: str):
    response = await delete_process(finetuned_model_name)
    return response


