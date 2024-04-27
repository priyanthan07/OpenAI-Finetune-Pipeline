from pydantic import BaseModel
import json

class FineTune(BaseModel):
    finetune_data: list
    class Config:
        extra = 'forbid'