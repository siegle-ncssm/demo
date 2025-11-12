"""
Tests for API serving
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.serving.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_list_models(self, client):
        """Test list models endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        assert "models" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
