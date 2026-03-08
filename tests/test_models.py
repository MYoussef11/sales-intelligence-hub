"""Tests for ML model classes."""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_lead_scorer_train(sample_leads_df):
    """Test that LeadScorer trains without error and returns metrics."""
    from ml_services.lead_scoring import LeadScorer

    scorer = LeadScorer()
    metrics = scorer.train(data=sample_leads_df)

    assert metrics is not None
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_score"] <= 1


def test_lead_scorer_predict(sample_leads_df):
    """Test that a trained LeadScorer can predict."""
    from ml_services.lead_scoring import LeadScorer

    scorer = LeadScorer()
    scorer.train(data=sample_leads_df)

    result = scorer.predict("website", 15)
    assert "conversion_probability" in result
    assert "risk_level" in result
    assert result["risk_level"] in ["High", "Low"]


def test_segmentation_train(sample_dealer_features_df):
    """Test that DealerSegmentation trains and returns metrics."""
    from ml_services.segmentation import DealerSegmentation

    segmentor = DealerSegmentation()
    metrics = segmentor.train_and_evaluate(data=sample_dealer_features_df)

    assert metrics is not None
    assert "silhouette_score" in metrics
    assert "n_clusters" in metrics
    assert metrics["n_clusters"] == 3


def test_segmentation_predict(sample_dealer_features_df):
    """Test that segmentation can assign clusters."""
    from ml_services.segmentation import DealerSegmentation

    segmentor = DealerSegmentation()
    segmentor.train_and_evaluate(data=sample_dealer_features_df)
    result = segmentor.get_segments(data=sample_dealer_features_df)

    assert result is not None
    assert len(result) == len(sample_dealer_features_df)
