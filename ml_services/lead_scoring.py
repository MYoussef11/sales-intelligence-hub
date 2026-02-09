import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
import pickle
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.schema import Lead

DATABASE_URL = "postgresql://admin:admin123@localhost:5432/sales_intelligence"

class LeadScorer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.encoder = LabelEncoder()
        
    def get_training_data(self):
        engine = create_engine(DATABASE_URL)
        query = """
        SELECT source, response_time_minutes, converted
        FROM leads
        """
        df = pd.read_sql(query, engine)
        return df

    def train(self):
        print("Training Lead Scoring Model...")
        df = self.get_training_data()
        
        if df.empty:
            return "No training data found"
            
        # Preprocessing
        df['source_encoded'] = self.encoder.fit_transform(df['source'])
        
        X = df[['source_encoded', 'response_time_minutes']]
        y = df['converted']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        print(f"Model Accuracy: {score:.2f}")
        
        # Save model
        with open('ml_services/lead_scorer.pkl', 'wb') as f:
            pickle.dump((self.model, self.encoder), f)
            
        return score

    def predict(self, source, response_time):
        # Load model if not in memory
        if not hasattr(self.model, 'estimators_'):
            try:
                with open('ml_services/lead_scorer.pkl', 'rb') as f:
                    self.model, self.encoder = pickle.load(f)
            except FileNotFoundError:
                return 0.0

        try:
            source_encoded = self.encoder.transform([source])[0]
        except ValueError:
            # Handle unknown source
            source_encoded = 0 
            
        prob = self.model.predict_proba([[source_encoded, response_time]])[0][1]
        return prob

if __name__ == "__main__":
    scorer = LeadScorer()
    scorer.train()
    print(f"Prediction for Website lead, 10 min response: {scorer.predict('website', 10)}")
