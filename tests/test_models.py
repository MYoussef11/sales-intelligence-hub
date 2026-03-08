"""Tests for ML model classes.

These tests mock database access so they run in CI without Postgres.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_leads_data():
    """Sample leads DataFrame matching the DB schema."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "source": np.random.choice(["website", "referral", "email", "trade_show", "cold_call"], n),
        "response_time_minutes": np.random.randint(1, 300, n),
        "converted": np.random.choice([True, False], n, p=[0.3, 0.7]),
    })


@pytest.fixture
def mock_dealer_data():
    """Sample dealer features matching what get_dealer_data() returns."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "dealer_id": range(1, n + 1),
        "avg_monthly_volume": np.random.uniform(10, 500, n),
        "churn_risk_score": np.random.uniform(0, 1, n),
    })


def test_lead_scorer_train(mock_leads_data):
    """Test that LeadScorer trains without error and returns metrics."""
    from ml_services.lead_scoring import LeadScorer

    scorer = LeadScorer()

    with patch.object(scorer, "get_training_data", return_value=mock_leads_data):
        metrics = scorer.train()

    assert metrics is not None
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_score"] <= 1


def test_lead_scorer_predict(mock_leads_data):
    """Test that a trained LeadScorer can predict."""
    from ml_services.lead_scoring import LeadScorer

    scorer = LeadScorer()

    with patch.object(scorer, "get_training_data", return_value=mock_leads_data):
        scorer.train()

    scorer.loaded = True
    result = scorer.predict("website", 15)

    # predict() returns a float probability
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_segmentation_train(mock_dealer_data):
    """Test that DealerSegmentation trains and returns metrics."""
    from ml_services.segmentation import DealerSegmentation

    segmentor = DealerSegmentation()

    with patch.object(segmentor, "get_dealer_data", return_value=mock_dealer_data):
        metrics = segmentor.train_and_evaluate()

    assert metrics is not None
    assert "silhouette_score" in metrics
    assert "n_clusters" in metrics
    assert metrics["n_clusters"] == 3
