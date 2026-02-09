import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
import pickle
import os
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class LeadScorer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.encoder = LabelEncoder()
        self.loaded = False
        
    def load_model(self):
        try:
            model_path = os.path.join(settings.MODELS_DIR, "lead_scorer.pkl")
            with open(model_path, 'rb') as f:
                self.model, self.encoder = pickle.load(f)
            self.loaded = True
            logger.info("Lead Scorer model loaded successfully.")
        except FileNotFoundError:
            logger.warning("Lead Scorer model not found. Please run training script.")
        
    def get_training_data(self):
        engine = create_engine(settings.DATABASE_URL)
        query = """
        SELECT source, response_time_minutes, converted
        FROM leads
        """
        df = pd.read_sql(query, engine)
        return df

    def train(self):
        logger.info("Training Lead Scoring Model...")
        df = self.get_training_data()
        
        if df.empty:
            logger.warning("No training data found")
            return 0.0
            
        # Preprocessing
        df['source_encoded'] = self.encoder.fit_transform(df['source'])
        
        X = df[['source_encoded', 'response_time_minutes']]
        y = df['converted']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        logger.info(f"Model Accuracy: {score:.2f}")
        
        return score

    def predict(self, source, response_time):
        if not self.loaded:
            self.load_model()
            
        try:
            if not hasattr(self.model, 'predict_proba'):
                 return 0.5 # Fallback
                 
            source_encoded = self.encoder.transform([source])[0]
            prob = self.model.predict_proba([[source_encoded, response_time]])[0][1]
            return prob
        except Exception as e:
            # Handle unknown source or other errors
            logger.error(f"Prediction error: {e}")
            return 0.0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scorer = LeadScorer()
    scorer.train()
    print(f"Prediction for Website lead, 10 min response: {scorer.predict('website', 10)}")
