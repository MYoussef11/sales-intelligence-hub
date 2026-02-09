import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.schema import Dealer

DATABASE_URL = "postgresql://admin:admin123@localhost:5432/sales_intelligence"

class DealerSegmentation:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=3, random_state=42) # Low, Med, High Value
        self.scaler = StandardScaler()
        
    def get_dealer_data(self):
        engine = create_engine(DATABASE_URL)
        query = """
        SELECT dealer_id, avg_monthly_volume, churn_risk_score
        FROM dealers
        """
        df = pd.read_sql(query, engine)
        return df

    def run_segmentation(self):
        print("Running Dealer Segmentation...")
        df = self.get_dealer_data()
        
        if df.empty:
            return None
            
        # Features
        X = df[['avg_monthly_volume', 'churn_risk_score']]
        X_scaled = self.scaler.fit_transform(X)
        
        df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Interpret clusters (Simplified)
        cluster_map = {
            0: "Standard",
            1: "High Value", 
            2: "At Risk"
        }
        # Note: In real app, we'd analyze centroids to name them dynamically
        
        result = df[['dealer_id', 'cluster']]
        return result

if __name__ == "__main__":
    segmentor = DealerSegmentation()
    segments = segmentor.run_segmentation()
    print(segments.head())
