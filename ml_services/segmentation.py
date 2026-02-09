import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import os
import sys
import pickle
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class DealerSegmentation:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=3, random_state=42) # Low, Med, High Value
        self.scaler = StandardScaler()
        self.loaded = False
        
    def load_model(self):
        try:
            model_path = os.path.join(settings.MODELS_DIR, "segmentation.pkl")
            with open(model_path, 'rb') as f:
                self.kmeans, self.scaler = pickle.load(f)
            self.loaded = True
            logger.info("Segmentation model loaded successfully.")
        except FileNotFoundError:
             logger.warning("Segmentation model not found. Running fresh segmentation.")

    def get_dealer_data(self):
        engine = create_engine(settings.DATABASE_URL)
        query = """
        SELECT dealer_id, avg_monthly_volume, churn_risk_score
        FROM dealers
        """
        df = pd.read_sql(query, engine)
        return df

    def run_segmentation(self):
        logger.info("Running Dealer Segmentation...")
        
        # Try to load model first
        if not self.loaded:
            self.load_model()
            
        df = self.get_dealer_data()
        
        if df.empty:
            return None
            
        # Features
        X = df[['avg_monthly_volume', 'churn_risk_score']]
        
        # If loaded, predict. If not, fit_predict (and ideally save, but checking mainly logic here)
        if self.loaded:
            X_scaled = self.scaler.transform(X)
            df['cluster'] = self.kmeans.predict(X_scaled)
        else:
            X_scaled = self.scaler.fit_transform(X)
            df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        result = df[['dealer_id', 'cluster']]
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    segmentor = DealerSegmentation()
    segments = segmentor.run_segmentation()
    if segments is not None:
        print(segments.head())
