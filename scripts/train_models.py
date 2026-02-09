import sys
import os
import pickle
import logging
import pandas as pd
from sqlalchemy import create_engine

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings
from ml_services.forecasting import train_forecast_model
from ml_services.lead_scoring import LeadScorer
from ml_services.segmentation import DealerSegmentation

settings = get_settings()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_save_lead_scorer():
    logger.info("Training Lead Scorer...")
    scorer = LeadScorer()
    try:
        accuracy = scorer.train()
        model_path = os.path.join(settings.MODELS_DIR, "lead_scorer.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump((scorer.model, scorer.encoder), f)
            
        logger.info(f"Lead Scorer saved to {model_path} (Accuracy: {accuracy})")
    except Exception as e:
        logger.error(f"Failed to train Lead Scorer: {e}")

def train_and_save_segmentation():
    logger.info("Training Dealer Segmentation...")
    segmentor = DealerSegmentation()
    try:
        # For segmentation, "training" is fitting the KMeans. 
        # In this POC, the run_segmentation method fits and predicts.
        # We will modify it to return the fitted model if needed, or just save the Cluster Centers.
        # For now, we'll save the fitted KMeans object if possible, but the current class structure
        # might need adjustment. Let's instantiated and fit.
        
        df = segmentor.get_dealer_data()
        if not df.empty:
            X = df[['avg_monthly_volume', 'churn_risk_score']]
            X_scaled = segmentor.scaler.fit_transform(X)
            segmentor.kmeans.fit(X_scaled)
            
            model_path = os.path.join(settings.MODELS_DIR, "segmentation.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump((segmentor.kmeans, segmentor.scaler), f)
                
            logger.info(f"Segmentation model saved to {model_path}")
        else:
            logger.warning("No dealer data for segmentation training.")
            
    except Exception as e:
        logger.error(f"Failed to train Segmentation: {e}")

# Note: Forecasting is per-dealer and usually on-demand or batch. 
# We don't save a single global model for forecasting in this architecture.

def main():
    logger.info("Starting Model Training Pipeline...")
    train_and_save_lead_scorer()
    train_and_save_segmentation()
    logger.info("Model Training Complete.")

if __name__ == "__main__":
    main()
