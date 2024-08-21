import joblib
import pandas as pd
from fastapi import FastAPI, Request
from Fraud import Fraud
from pydantic import BaseModel
from typing import List, Dict
import os
# loading model
model = joblib.load('./models/model_cycle1.joblib')

# initialize API
app = FastAPI()

class DataModel(BaseModel):
    step: int
    type: str
    amount: float
    name_orig: str
    oldbalance_org: float
    newbalance_orig: float
    name_dest: str
    oldbalance_dest: float
    newbalance_dest: float
    isFraud: int
@app.get("/hello")
def SayHello():
    return {"msg": "hello"}


@app.post("/fraud/predict")
async def churn_predict(request: Request):
    test_json = await request.json()
   
   
    if test_json: # there is data
        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame([test_json])
        else: # multiple examples
            test_raw = pd.DataFrame(test_json)
            
        # Instantiate Fraud class
        pipeline = Fraud()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
        
    else:
        return {"message": "No data provided"}

if __name__ == "__main__":
   
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
