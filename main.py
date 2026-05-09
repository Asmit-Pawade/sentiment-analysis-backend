from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from model import predict_sentiment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ADD THIS (CRITICAL)
@app.get("/health")
def health_check():
    return {"status": "ok"}

class InputText(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

@app.post("/predict")
def predict(data: InputText):
    try:
        return predict_sentiment(data.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
def batch_predict(data: BatchInput):
    return [predict_sentiment(text) for text in data.texts]