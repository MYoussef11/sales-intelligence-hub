from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import sys
import os
import uvicorn
import logging
import httpx
from sqlalchemy import create_engine, text

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

settings = get_settings()

# --- Sentry Error Tracking ---
import sentry_sdk
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=1.0,
        send_default_pii=False,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        release=settings.APP_VERSION,
    )
    logging.getLogger(__name__).info("Sentry error tracking enabled")

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ml_services.forecasting import train_forecast_model
from ml_services.lead_scoring import LeadScorer
from ml_services.segmentation import DealerSegmentation
from ml_services.orchestrator import run_chat

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# Initialize Services
lead_scorer = LeadScorer()
segmentor = DealerSegmentation()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Sales Intelligence Hub API...")
    try:
        lead_scorer.load_model()
        segmentor.load_model()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

# --- Request Models ---

class LeadRequest(BaseModel):
    source: str
    response_time_minutes: int

class AgentQuery(BaseModel):
    question: str

class NewLeadRequest(BaseModel):
    contact_name: str
    email: str
    company: Optional[str] = None
    source: str = "website"
    inquiry_text: Optional[str] = None

# --- Existing Endpoints ---

@app.get("/")
def read_root():
    return {"status": "Sales Intelligence Hub API is running"}

@app.get("/forecast/{dealer_id}")
def get_forecast(dealer_id: int):
    try:
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
        result = run_chat(query.question)
        return result  # Now returns {"answer": ..., "agent_type": "sql"|"rag"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- New Endpoints ---

@app.get("/analytics/summary")
def analytics_summary():
    """Live KPI summary for the dashboard and n8n alerts."""
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            total_dealers = conn.execute(text("SELECT COUNT(*) FROM dealers")).scalar()
            total_leads = conn.execute(text("SELECT COUNT(*) FROM leads")).scalar()
            
            revenue_row = conn.execute(text("""
                SELECT 
                    COALESCE(SUM(CASE WHEN date >= CURRENT_DATE - INTERVAL '30 days' THEN sale_price END), 0) as current_month,
                    COALESCE(SUM(CASE WHEN date >= CURRENT_DATE - INTERVAL '60 days' AND date < CURRENT_DATE - INTERVAL '30 days' THEN sale_price END), 0) as prev_month
                FROM transactions
            """)).fetchone()
            
            current_revenue = float(revenue_row[0])
            prev_revenue = float(revenue_row[1])
            revenue_change_pct = round(((current_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0, 1)
            
            active_deals = conn.execute(text(
                "SELECT COUNT(*) FROM transactions WHERE date >= CURRENT_DATE - INTERVAL '30 days'"
            )).scalar()
            
            low_score_leads = conn.execute(text(
                "SELECT COUNT(*) FROM leads WHERE converted = false"
            )).scalar()
        
        return {
            "total_dealers": total_dealers,
            "total_leads": total_leads,
            "total_revenue": current_revenue,
            "prev_revenue": prev_revenue,
            "revenue_change_pct": revenue_change_pct,
            "active_deals": active_deals,
            "low_score_leads": low_score_leads,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dealers")
def list_dealers():
    """List all dealers for dropdown filters."""
    try:
        engine = create_engine(settings.DATABASE_URL)
        df = pd.read_sql("SELECT dealer_id, name as dealer_name FROM dealers ORDER BY dealer_id", engine)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leads")
async def create_lead(lead: NewLeadRequest):
    """Create a new lead from the landing page and trigger n8n."""
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            # Auto-assign a dealer (round-robin based on lead count)
            dealer_row = conn.execute(text(
                "SELECT dealer_id FROM dealers ORDER BY dealer_id OFFSET (SELECT COUNT(*) FROM leads) % (SELECT COUNT(*) FROM dealers) LIMIT 1"
            )).fetchone()
            dealer_id = dealer_row[0] if dealer_row else 1

            # Pack contact info into inquiry_text
            full_inquiry = f"Contact: {lead.contact_name} | Email: {lead.email}"
            if lead.company:
                full_inquiry += f" | Company: {lead.company}"
            if lead.inquiry_text:
                full_inquiry += f"\n{lead.inquiry_text}"

            result = conn.execute(
                text("""
                    INSERT INTO leads (dealer_id, source, inquiry_text, response_time_minutes, converted, created_at)
                    VALUES (:dealer_id, :source, :inquiry_text, 0, false, NOW())
                    RETURNING lead_id
                """),
                {
                    "dealer_id": dealer_id,
                    "source": lead.source,
                    "inquiry_text": full_inquiry,
                }
            )
            conn.commit()
            lead_id = result.scalar()
        
        # Trigger n8n webhook asynchronously (non-blocking)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    "http://n8n:5678/webhook/new-lead",
                    json={"lead_id": lead_id}
                )
                logger.info(f"n8n webhook triggered for lead {lead_id}")
        except Exception as webhook_err:
            logger.warning(f"n8n webhook skipped (not available): {webhook_err}")
        
        return {"id": lead_id, "status": "created", "message": f"Lead #{lead_id} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Landing Page ---

@app.get("/landing", response_class=HTMLResponse)
def landing_page():
    """Serve the lead intake landing page."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "landing.html")
    try:
        with open(template_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Landing page template not found</h1>", status_code=404)

# --- Admin Endpoints (internal use by n8n workflows) ---

@app.post("/admin/retrain")
async def retrain_models():
    """Trigger model retraining pipeline. Returns training metrics."""
    try:
        from scripts.train_models import main as run_training
        results = run_training()
        
        summary = {
            "status": "completed",
            "lead_scorer": {"status": "ok", "metrics": results.get("lead_scorer")} if results.get("lead_scorer") else {"status": "failed"},
            "segmentation": {"status": "ok", "metrics": results.get("segmentation")} if results.get("segmentation") else {"status": "failed"},
            "forecasting": {"status": "ok"} if results.get("forecasting") else {"status": "failed"},
        }
        
        # Reload models in memory
        try:
            lead_scorer.load_model()
            segmentor.load_model()
            logger.info("Models reloaded after retraining")
        except Exception as reload_err:
            logger.warning(f"Model reload after retrain: {reload_err}")
        
        return summary
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/drift-check")
async def check_drift(table: str = "leads"):
    """Run data drift analysis. Returns drift summary and saves HTML report."""
    try:
        from ml_services.drift_monitor import DriftMonitor
        monitor = DriftMonitor()
        result = monitor.run_drift_check(table)
        return result
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
