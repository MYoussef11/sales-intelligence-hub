"""Shared test fixtures for Sales Intelligence Hub."""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set required env vars BEFORE importing anything that uses config
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DB", "test_db")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")


@pytest.fixture
def sample_leads_df():
    """Sample leads DataFrame matching the DB schema."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "lead_id": range(1, n + 1),
        "source": np.random.choice(["website", "referral", "email", "trade_show", "cold_call"], n),
        "response_time_minutes": np.random.randint(1, 300, n),
        "converted": np.random.choice([True, False], n, p=[0.3, 0.7]),
    })


@pytest.fixture
def sample_transactions_df():
    """Sample transactions DataFrame for forecasting."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    return pd.DataFrame({
        "dealer_id": 1,
        "sale_date": dates,
        "total_amount": np.random.uniform(5000, 50000, len(dates)),
    })


@pytest.fixture
def sample_dealer_features_df():
    """Sample dealer features DataFrame for segmentation."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "dealer_id": range(1, n + 1),
        "total_revenue": np.random.uniform(100000, 5000000, n),
        "total_transactions": np.random.randint(10, 500, n),
        "avg_transaction": np.random.uniform(5000, 50000, n),
        "active_months": np.random.randint(1, 36, n),
    })
