"""Step 4 - A small web API: send customer details, get churn probability + advice."""
import json, os, joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "outputs/model.joblib")
THRESHOLD = json.load(open(os.getenv("METRICS_PATH", "outputs/metrics.json")))["chosen_threshold"]
model = joblib.load(MODEL_PATH)
app = FastAPI(title="Churn Prediction API")

class Customer(BaseModel):
    gender: str; SeniorCitizen: int; Partner: str; Dependents: str; tenure: int
    PhoneService: str; InternetService: str; OnlineSecurity: str; TechSupport: str
    Contract: str; PaperlessBilling: str; PaymentMethod: str
    MonthlyCharges: float; TotalCharges: float; support_calls: int

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(c: Customer):
    p = float(model.predict_proba(pd.DataFrame([c.model_dump()]))[0, 1])
    risky = p >= THRESHOLD
    return {"churn_probability": round(p, 3), "high_risk": risky,
            "action": "Send retention offer" if risky else "No action needed"}
