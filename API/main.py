from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import pred_inputs
from predict import predict

app = FastAPI()

# ➜ 允許所有來源（讓所有人都能呼叫 API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def pred(pred_inputs:pred_inputs):
    preds = predict(pred_inputs.dict())
    return preds 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
