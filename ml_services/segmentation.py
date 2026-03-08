import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
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

    def train_and_evaluate(self):
        """Fit segmentation model and return evaluation metrics."""
        logger.info("Training Dealer Segmentation...")
        df = self.get_dealer_data()
        
        if df.empty:
            logger.warning("No dealer data for segmentation.")
            return None
        
        X = df[['avg_monthly_volume', 'churn_risk_score']]
        X_scaled = self.scaler.fit_transform(X)
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Evaluation Metrics
        sil_score = silhouette_score(X_scaled, labels)
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        inertia = self.kmeans.inertia_
        
        metrics = {
            "silhouette_score": sil_score,
            "inertia": inertia,
            "n_clusters": self.kmeans.n_clusters,
            "total_dealers": len(df),
            "cluster_distribution": cluster_counts.to_dict(),
        }
        
        logger.info("=" * 50)
        logger.info("DEALER SEGMENTATION — Evaluation Results")
        logger.info("=" * 50)
        logger.info(f"  Silhouette Score: {sil_score:.4f}  (1.0 = perfect, 0.0 = overlapping)")
        logger.info(f"  Inertia (SSE):    {inertia:.2f}")
        logger.info(f"  Clusters:         {self.kmeans.n_clusters}")
        logger.info(f"  Total Dealers:    {len(df)}")
        logger.info("  Cluster Distribution:")
        for cluster_id, count in cluster_counts.items():
            pct = (count / len(df)) * 100
            logger.info(f"    Cluster {cluster_id}: {count} dealers ({pct:.1f}%)")
        logger.info("-" * 50)
        
        return metrics

    def run_segmentation(self):
        logger.info("Running Dealer Segmentation...")
        
        if not self.loaded:
            self.load_model()
            
        df = self.get_dealer_data()
        
        if df.empty:
            return None
            
        X = df[['avg_monthly_volume', 'churn_risk_score']]
        
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
    segmentor.train_and_evaluate()
    segments = segmentor.run_segmentation()
    if segments is not None:
        print(segments.head())
