from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
import sys
import os
import uvicorn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.forecasting import train_forecast_model # In prod, load saved model
from ml_services.lead_scoring import LeadScorer
from ml_services.segmentation import DealerSegmentation
from ml_services.rag_agent import InternalSalesAgent

app = FastAPI(title="Sales Intelligence Hub API", version="1.0")

# Initialize Services (Load models)
# In a real app, we'd load pickled models here to avoid retraining on every request
lead_scorer = LeadScorer()
segmentor = DealerSegmentation()
rag_agent = InternalSalesAgent()

# Pre-train/load for POC
print("Loading/Training models...")
lead_scorer.train() 
# segmentor.train() # Segmentation is usually batch, not real-time 

# Request Models
class LeadRequest(BaseModel):
    source: str
    response_time_minutes: int

class AgentQuery(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "Sales Intelligence Hub API is running"}

@app.get("/forecast/{dealer_id}")
def get_forecast(dealer_id: int):
    try:
        # For POC, we run the forecast on request. In prod, we'd fetch pre-calculated.
        forecast, status = train_forecast_model(dealer_id)
        if forecast is None:
            raise HTTPException(status_code=404, detail=status)
        
        return forecast.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score_lead")
def score_lead(lead: LeadRequest):
    try:
        prob = lead_scorer.predict(lead.source, lead.response_time_minutes)
        return {"conversion_probability": prob, "risk_level": "High" if prob < 0.3 else "Low"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/segments")
def get_segments():
    try:
        result = segmentor.run_segmentation()
        if result is None:
             raise HTTPException(status_code=404, detail="No dealer data found")
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/query")
def query_agent(query: AgentQuery):
    try:
        # Mock ingestion for POC context
        policy_content = """
        Return Policy: All vehicles can be returned within 14 days if under 500km usage.
        Warranty: Standard warranty is 2 years for engine and transmission.
        Discount: Sales reps can authorize up to 5% discount. Managers up to 10%.
        """
        rag_agent.ingest_policies(policy_content)
        answer = rag_agent.query(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
