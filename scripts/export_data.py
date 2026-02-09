import pandas as pd
from sqlalchemy import create_engine
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_settings()

def export_table_to_csv(table_name, engine):
    """
    Exports a single table to a CSV file in the DATA_DIR.
    """
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        
        output_path = os.path.join(settings.DATA_DIR, f"{table_name}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully exported {table_name} to {output_path} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Failed to export {table_name}: {e}")

def main():
    logger.info("Starting database export...")
    
    try:
        engine = create_engine(settings.DATABASE_URL)
        tables = ["dealers", "inventory", "transactions", "leads", "employees"]
        
        for table in tables:
            export_table_to_csv(table, engine)
            
        logger.info("Export complete.")
    except Exception as e:
        logger.critical(f"Database connection failed: {e}")

if __name__ == "__main__":
    main()
