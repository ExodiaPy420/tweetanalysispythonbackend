"""
Unit tests for the sentiment analysis API.
"""
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_check_returns_200(self):
        """Health check should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_has_correct_fields(self):
        """Health check should return expected fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "data_loaded" in data
        assert "total_tweets" in data
    
    def test_health_check_status_healthy(self):
        """Health check status should be 'healthy'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        # Model and data will be loaded on first health check
        assert data["data_loaded"] is True
        assert data["total_tweets"] > 0


class TestAnalyzeEndpoint:
    """Tests for the /analyze endpoint."""
    
    def test_analyze_with_valid_query_returns_200(self):
        """Valid query should return 200 OK."""
        response = client.get("/analyze?query=flight")
        assert response.status_code == 200
    
    def test_analyze_response_has_correct_structure(self):
        """Response should have expected fields."""
        response = client.get("/analyze?query=flight")
        data = response.json()
        assert "query" in data
        assert "total_tweets_found" in data
        assert "results" in data
        assert "tweets" in data
    
    def test_analyze_results_has_sentiment_counts(self):
        """Results should contain sentiment counts."""
        response = client.get("/analyze?query=flight")
        data = response.json()
        results = data["results"]
        assert "positive" in results
        assert "negative" in results
        assert "neutral" in results
    
    def test_analyze_empty_query_returns_400(self):
        """Empty query should return 400 Bad Request."""
        response = client.get("/analyze?query=")
        assert response.status_code == 400
    
    def test_analyze_whitespace_query_returns_400(self):
        """Whitespace-only query should return 400."""
        response = client.get("/analyze?query=   ")
        assert response.status_code == 400
    
    def test_analyze_sql_injection_attempt_returns_400(self):
        """SQL injection attempts should be rejected."""
        dangerous_queries = [
            "DROP TABLE",
            "DELETE FROM",
            "'; DROP TABLE--",
            "/* comment */"
        ]
        for query in dangerous_queries:
            response = client.get(f"/analyze?query={query}")
            assert response.status_code == 400, f"Failed to block: {query}"
    
    def test_analyze_nonexistent_term_returns_empty_results(self):
        """Non-existent term should return zero results."""
        response = client.get("/analyze?query=xyzabc123notfound")
        assert response.status_code == 200
        data = response.json()
        assert data["total_tweets_found"] == 0
        assert len(data["tweets"]) == 0
    
    def test_analyze_query_is_case_insensitive(self):
        """Query should be case-insensitive."""
        response1 = client.get("/analyze?query=FLIGHT")
        response2 = client.get("/analyze?query=flight")
        assert response1.status_code == 200
        assert response2.status_code == 200
        # Both should find results
        assert response1.json()["total_tweets_found"] > 0
        assert response2.json()["total_tweets_found"] > 0
    
    def test_analyze_tweets_have_sentiment_field(self):
        """Each tweet should have a sentiment field."""
        response = client.get("/analyze?query=flight")
        data = response.json()
        if len(data["tweets"]) > 0:
            for tweet in data["tweets"]:
                assert "sentiment" in tweet
                assert tweet["sentiment"] in ["positive", "negative", "neutral"]
    
    def test_analyze_max_query_length(self):
        """Query longer than 100 characters should be rejected."""
        long_query = "a" * 101
        response = client.get(f"/analyze?query={long_query}")
        assert response.status_code == 400


class TestRateLimiting:
    """Tests for rate limiting functionality."""
    
    def test_rate_limit_allows_normal_usage(self):
        """Normal usage should not trigger rate limit."""
        # Make 5 requests (well under the 60/minute limit)
        for _ in range(5):
            response = client.get("/analyze?query=test")
            assert response.status_code == 200
    
    # Note: Testing actual rate limit (61 requests) would slow down tests
    # In production, this would be tested separately
