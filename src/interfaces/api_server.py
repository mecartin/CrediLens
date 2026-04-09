from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="CrediLens API",
    version="4.0.0",
    description="AI Credit Risk Analysis API"
)

def load_system():
    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / 'models' / 'saved_models' / 'xgb_model.pkl'
    prep_path = base_path / 'models' / 'saved_models' / 'preprocessor.pkl'
    if model_path.exists() and prep_path.exists():
        return joblib.load(model_path), joblib.load(prep_path)
    return None, None

class PredictionRequest(BaseModel):
    loan_amnt: float
    term: int
    int_rate: float
    installment: float
    emp_length: float
    home_ownership: str
    annual_inc: float
    dti: float
    fico: float
    revol_bal: float
    revol_util: float
    pub_rec_bankruptcies: float
    explain: bool = True
    generate_counterfactuals: bool = False
    generate_recourse: bool = False

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    decision: str
    risk_score: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        model, preprocessor = load_system()
        if not model:
            raise HTTPException(status_code=500, detail="Models not trained.")
            
        # Format payload analogous to simple UI input
        df = pd.DataFrame([{
            'loan_amnt': request.loan_amnt,
            'term': request.term,
            'int_rate': request.int_rate,
            'installment': request.installment,
            'grade': 'B', 
            'sub_grade': 'B1', 
            'emp_title': 'General', 
            'emp_length': request.emp_length,
            'home_ownership': request.home_ownership,
            'annual_inc': request.annual_inc,
            'verification_status': 'Verified',
            'purpose': 'debt_consolidation',
            'title': 'Debt consolidation',
            'zip_code': '000xx',
            'addr_state': 'CA',
            'dti': request.dti,
            'fico_range_low': request.fico,
            'fico_range_high': request.fico + 4,
            'open_acc': 5,
            'pub_rec': 0,
            'revol_bal': request.revol_bal,
            'revol_util': request.revol_util,
            'total_acc': 10,
            'application_type': 'Individual',
            'mort_acc': 1 if request.home_ownership == 'MORTGAGE' else 0,
            'pub_rec_bankruptcies': request.pub_rec_bankruptcies
        }])
        
        processed = preprocessor.transform(df)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]
        
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'decision': 'approved' if prediction == 0 else 'denied',
            'risk_score': float(probability)
        }
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "4.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
