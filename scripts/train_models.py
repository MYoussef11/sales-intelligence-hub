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
    logger.info("=" * 60)
    logger.info("PHASE 1: Lead Scorer")
    logger.info("=" * 60)
    scorer = LeadScorer()
    try:
        metrics = scorer.train()
        model_path = os.path.join(settings.MODELS_DIR, "lead_scorer.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump((scorer.model, scorer.encoder), f)
            
        logger.info(f"Lead Scorer saved to {model_path}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to train Lead Scorer: {e}")
        return None

def train_and_save_segmentation():
    logger.info("=" * 60)
    logger.info("PHASE 2: Dealer Segmentation")
    logger.info("=" * 60)
    segmentor = DealerSegmentation()
    try:
        metrics = segmentor.train_and_evaluate()
        
        if metrics:
            model_path = os.path.join(settings.MODELS_DIR, "segmentation.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump((segmentor.kmeans, segmentor.scaler), f)
            logger.info(f"Segmentation model saved to {model_path}")
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to train Segmentation: {e}")
        return None

def train_and_evaluate_forecasting():
    logger.info("=" * 60)
    logger.info("PHASE 3: Revenue Forecasting (Sample Dealer)")
    logger.info("=" * 60)
    try:
        # Train and evaluate on a sample dealer to log metrics
        forecast, status = train_forecast_model(dealer_id=1)
        if forecast is not None:
            logger.info(f"Sample 30-day forecast generated successfully ({len(forecast)} days)")
        else:
            logger.warning(f"Forecasting evaluation: {status}")
        return forecast is not None
    except Exception as e:
        logger.error(f"Failed to evaluate Forecasting: {e}")
        return False

def main():
    logger.info("=" * 60)
    logger.info("  SALES INTELLIGENCE HUB — Model Training Pipeline")
    logger.info("=" * 60)
    
    results = {}
    
    results['lead_scorer'] = train_and_save_lead_scorer()
    results['segmentation'] = train_and_save_segmentation()
    results['forecasting'] = train_and_evaluate_forecasting()
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAINING PIPELINE COMPLETE — Summary")
    logger.info("=" * 60)
    
    if results['lead_scorer']:
        logger.info(f"  ✅ Lead Scorer: Accuracy={results['lead_scorer']['accuracy']:.4f}, F1={results['lead_scorer']['f1_score']:.4f}")
    else:
        logger.info("  ❌ Lead Scorer: Failed")
    
    if results['segmentation']:
        logger.info(f"  ✅ Segmentation: Silhouette={results['segmentation']['silhouette_score']:.4f}")
    else:
        logger.info("  ❌ Segmentation: Failed")
    
    if results['forecasting']:
        logger.info("  ✅ Forecasting: Evaluated (see RMSE/MAE above)")
    else:
        logger.info("  ❌ Forecasting: Failed")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
