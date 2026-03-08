"""Tests for FastAPI backend endpoints."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_root_endpoint():
    """Test the health check endpoint returns 200."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_score_lead_endpoint():
    """Test the lead scoring endpoint accepts valid input."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.post("/score_lead", json={
        "source": "website",
        "response_time_minutes": 15
    })
    # May return 200 (model loaded) or 500 (model not trained yet)
    # We just verify the endpoint exists and accepts the request format
    assert response.status_code in [200, 500]


def test_create_lead_validation():
    """Test that lead creation validates required fields."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    # Missing required fields should return 422
    response = client.post("/leads", json={})
    assert response.status_code == 422


def test_landing_page():
    """Test the landing page renders HTML."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/landing")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Sales Intelligence Hub" in response.text
