from fastapi import FastAPI
from model import pred_inputs
from predict import predict

app = FastAPI()

@app.post("/predict/")
async def pred(pred_inputs:pred_inputs):
    preds = predict(pred_inputs.dict())
    return preds 