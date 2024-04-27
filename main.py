from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
from routes import router as fine_tune_router

app = FastAPI()

app.include_router(fine_tune_router)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:4000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run("main:app", host='localhost', port= 8000, reload=True)