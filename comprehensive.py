import numpy as np
#!/usr/bin/env python3
"""
Comprehensive test suite for Legal AI API
Tests all endpoints with various scenarios
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
import json
from datetime import datetime
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
API_VERSION = "v1"

# Test data fixtures
@pytest.fixture
def quantum_analysis_data():
    return {
        "case_type": "employment",
        "description": "Wrongful termination case",
        "jurisdiction": "NSW",
        "arguments": [
            "Employee was terminated without cause",
            "Performance reviews were consistently excellent",
            "Termination occurred after whistleblowing"
        ],
        "precedents": ["Smith v ABC Corp", "Jones v XYZ Ltd"]
    }

@pytest.fixture
def precedent_analysis_data():
    return {
        "case_description": "Employment discrimination based on age",
        "relevant_acts": ["Fair Work Act 2009", "Age Discrimination Act 2004"],
        "year_range": [2018, 2023],
        "limit": 10
    }

@pytest.fixture
def settlement_analysis_data():
    return {
        "case_type": "personal_injury",
        "claim_amount": 250000.0,
        "injury_severity": "moderate",
        "liability_admission": True,
        "negotiation_stage": "mediation"
    }

@pytest.fixture
def prediction_data():
    return {
        "case_data": {
            "case_type": "employment",
            "strength_score": 75,
            "precedent_support": 80,
            "jurisdiction": "NSW"
        },
        "prediction_type": "outcome",
        "confidence_required": 0.7
    }

@pytest.fixture
def strategy_data():
    return {
        "case_summary": "Complex employment dispute involving wrongful termination and discrimination",
        "objectives": [
            "Maximize compensation",
            "Achieve quick resolution",
            "Avoid publicity"
        ],
        "constraints": {
            "budget": 50000,
            "timeline": "6_months"
        },
        "risk_tolerance": "medium"
    }

@pytest.fixture
def search_data():
    return {
        "query": "wrongful termination compensation NSW",
        "search_type": "hybrid",
        "filters": {
            "jurisdiction": "NSW",
            "year_from": 2020
        },
        "limit": 20
    }

# Base test class
class TestLegalAIAPI:
    """Base test class with common functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client"""
        from app.main import app
        self.client = TestClient(app)
        
    def assert_valid_response(self, response, expected_status=200):
        """Assert response is valid"""
        assert response.status_code == expected_status
        if expected_status == 200:
            data = response.json()
            assert data.get("success", True) == True
            assert "timestamp" in data
        return response.json()

# Test General Endpoints
class TestGeneralEndpoints(TestLegalAIAPI):
    """Test general endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        data = self.assert_valid_response(response)
        
        assert data["name"] == "Australian Legal AI API"
        assert "version" in data
        assert "endpoints" in data
        
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        data = self.assert_valid_response(response)
        
        assert data["status"] == "healthy"
        assert "services" in data
        assert "corpus_stats" in data

# Test Analysis Endpoints
class TestAnalysisEndpoints(TestLegalAIAPI):
    """Test analysis endpoints"""
    
    def test_quantum_analysis(self, quantum_analysis_data):
        """Test quantum analysis endpoint"""
        response = self.client.post(
            f"/api/{API_VERSION}/analysis/quantum",
            json=quantum_analysis_data
        )
        data = self.assert_valid_response(response)
        
        assert data["analysis_type"] == "quantum_prediction"
        assert "results" in data
        assert 0 <= data["confidence"] <= 1
        
    def test_quantum_analysis_invalid_data(self):
        """Test quantum analysis with invalid data"""
        response = self.client.post(
            f"/api/{API_VERSION}/analysis/quantum",
            json={"invalid": "data"}
        )
        assert response.status_code == 422
        
    def test_precedent_analysis(self, precedent_analysis_data):
        """Test precedent analysis endpoint"""
        response = self.client.post(
            f"/api/{API_VERSION}/analysis/precedents",
            json=precedent_analysis_data
        )
        data = self.assert_valid_response(response)
        
        assert data["analysis_type"] == "precedent_analysis"
        assert "citations" in data
        
    def test_settlement_analysis(self, settlement_analysis_data):
        """Test settlement analysis endpoint"""
        response = self.client.post(
            f"/api/{API_VERSION}/analysis/settlement",
            json=settlement_analysis_data
        )
        data = self.assert_valid_response(response)
        
        assert data["analysis_type"] == "settlement_optimization"
        assert "results" in data
        
    def test_argument_analysis(self):
        """Test argument strength analysis"""
        data = {
            "case_type": "employment",
            "description": "Employee claims constructive dismissal after workplace changes",
            "jurisdiction": "VIC"
        }
        response = self.client.post(
            f"/api/{API_VERSION}/analysis/arguments",
            json=data
        )
        result = self.assert_valid_response(response)
        
        assert result["analysis_type"] == "argument_strength"
        assert "results" in result

# Test Prediction Endpoints
class TestPredictionEndpoints(TestLegalAIAPI):
    """Test prediction endpoints"""
    
    def test_monte_carlo_simulation(self, prediction_data):
        """Test Monte Carlo simulation endpoint"""
        response = self.client.post(
            f"/api/{API_VERSION}/prediction/simulate",
            json=prediction_data
        )
        data = self.assert_valid_response(response)
        
        assert data["prediction_type"] == "monte_carlo_simulation"
        assert "prediction" in data
        assert "alternatives" in data
        
    def test_outcome_prediction(self, prediction_data):
        """Test outcome prediction endpoint"""
        response = self.client.post(
            f"/api/{API_VERSION}/prediction/outcome",
            json=prediction_data
        )
        data = self.assert_valid_response(response)
        
        assert data["prediction_type"] == "outcome_prediction"
        assert 0 <= data["confidence"] <= 1
        
    def test_timeline_prediction(self, prediction_data):
        """Test timeline prediction endpoint"""
        response = self.client.post(
            f"/api/{API_VERSION}/prediction/timeline",
            json=prediction_data
        )
        data = self.assert_valid_response(response)
        
        assert data["prediction_type"] == "timeline_prediction"
        assert "prediction" in data

# Test Strategy Endpoints
class TestStrategyEndpoints(TestLegalAIAPI):
    """Test strategy endpoints"""
    
    def test_generate_strategy(self, strategy_data):
        """Test strategy generation endpoint"""
        response = self.client.post(
            f"/api/{API_VERSION}/strategy/generate",
            json=strategy_data
        )
        data = self.assert_valid_response(response)
        
        assert "strategies" in data
        assert "recommended_strategy" in data
        assert "risk_assessment" in data
        
    def test_adapt_strategy(self, strategy_data):
        """Test strategy adaptation endpoint"""
        strategy_data["metadata"] = {
            "current_strategy": {"approach": "negotiation"},
            "new_information": {"opponent_weakness": "time_pressure"}
        }
        
        response = self.client.post(
            f"/api/{API_VERSION}/strategy/adapt",
            json=strategy_data
        )
        data = self.assert_valid_response(response)
        
        assert "strategies" in data
        assert "recommended_strategy" in data

# Test Search Endpoints
class TestSearchEndpoints(TestLegalAIAPI):
    """Test search endpoints"""
    
    def test_search_cases(self, search_data):
        """Test case search endpoint"""
        response = self.client.post(
            f"/api/{API_VERSION}/search/cases",
            json=search_data
        )
        data = self.assert_valid_response(response)
        
        assert data["query"] == search_data["query"]
        assert "total_results" in data
        assert "results" in data
        
    def test_search_legislation(self):
        """Test legislation search endpoint"""
        data = {
            "query": "Fair Work Act 2009 unfair dismissal",
            "search_type": "keyword",
            "limit": 5
        }
        
        response = self.client.post(
            f"/api/{API_VERSION}/search/legislation",
            json=data
        )
        result = self.assert_valid_response(response)
        
        assert "results" in result

# Test Admin Endpoints
class TestAdminEndpoints(TestLegalAIAPI):
    """Test admin endpoints"""
    
    def test_get_stats(self):
        """Test stats endpoint"""
        response = self.client.get(f"/api/{API_VERSION}/admin/stats")
        data = self.assert_valid_response(response)
        
        assert "cases" in data
        assert "settlements" in data
        assert "precedents" in data
        
    def test_clear_cache(self):
        """Test cache clear endpoint"""
        response = self.client.post(f"/api/{API_VERSION}/admin/cache/clear")
        data = self.assert_valid_response(response)
        
        assert data["success"] == True

# Test WebSocket Endpoint
class TestWebSocketEndpoint:
    """Test WebSocket functionality"""
    
    @pytest.mark.asyncio
    async def test_websocket_assistant(self):
        """Test WebSocket legal assistant"""
        from fastapi.testclient import TestClient
        from app.main import app
        
        with TestClient(app) as client:
            with client.websocket_connect("/ws/assistant") as websocket:
                # Send a query
                websocket.send_json({
                    "query": "What is unfair dismissal?",
                    "context": {"jurisdiction": "NSW"}
                })
                
                # Receive response
                data = websocket.receive_json()
                assert data["type"] == "response"
                assert "data" in data
                assert "timestamp" in data

# Integration Tests
class TestIntegration(TestLegalAIAPI):
    """Integration tests for complete workflows"""
    
    def test_complete_case_analysis_workflow(self):
        """Test complete case analysis workflow"""
        
        # 1. Search for relevant cases
        search_response = self.client.post(
            f"/api/{API_VERSION}/search/cases",
            json={
                "query": "employment discrimination age",
                "search_type": "semantic",
                "limit": 5
            }
        )
        search_data = self.assert_valid_response(search_response)
        
        # 2. Analyze precedents
        precedent_response = self.client.post(
            f"/api/{API_VERSION}/analysis/precedents",
            json={
                "case_description": "Age discrimination in employment",
                "relevant_acts": ["Age Discrimination Act 2004"],
                "limit": 5
            }
        )
        precedent_data = self.assert_valid_response(precedent_response)
        
        # 3. Perform quantum analysis
        quantum_response = self.client.post(
            f"/api/{API_VERSION}/analysis/quantum",
            json={
                "case_type": "employment",
                "description": "Age discrimination case",
                "jurisdiction": "NSW",
                "arguments": ["Clear evidence of age-based decisions"],
                "precedents": []
            }
        )
        quantum_data = self.assert_valid_response(quantum_response)
        
        # 4. Generate strategy
        strategy_response = self.client.post(
            f"/api/{API_VERSION}/strategy/generate",
            json={
                "case_summary": "Age discrimination in employment with strong evidence",
                "objectives": ["Fair compensation", "Policy change"],
                "risk_tolerance": "medium"
            }
        )
        strategy_data = self.assert_valid_response(strategy_response)
        
        # Verify workflow continuity
        assert search_data["total_results"] >= 0
        assert precedent_data["analysis_type"] == "precedent_analysis"
        assert quantum_data["confidence"] > 0
        assert len(strategy_data["strategies"]) > 0

# Performance Tests
class TestPerformance(TestLegalAIAPI):
    """Performance and load tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, quantum_analysis_data):
        """Test handling of concurrent requests"""
        import httpx
        
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            # Send 10 concurrent requests
            tasks = []
            for i in range(10):
                task = client.post(
                    f"/api/{API_VERSION}/analysis/quantum",
                    json=quantum_analysis_data
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All should succeed
            for response in responses:
                assert response.status_code == 200
                
    def test_large_search_request(self):
        """Test handling of large search requests"""
        response = self.client.post(
            f"/api/{API_VERSION}/search/cases",
            json={
                "query": "employment " * 50,  # Long query
                "search_type": "hybrid",
                "limit": 100
            }
        )
        # Should handle gracefully
        assert response.status_code in [200, 422]

# Error Handling Tests
class TestErrorHandling(TestLegalAIAPI):
    """Test error handling"""
    
    def test_404_endpoint(self):
        """Test 404 handling"""
        response = self.client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
    def test_invalid_json(self):
        """Test invalid JSON handling"""
        response = self.client.post(
            f"/api/{API_VERSION}/analysis/quantum",
            data="invalid json"
        )
        assert response.status_code == 422
        
    def test_missing_required_fields(self):
        """Test missing required fields"""
        response = self.client.post(
            f"/api/{API_VERSION}/analysis/quantum",
            json={"case_type": "employment"}  # Missing required fields
        )
        assert response.status_code == 422

# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])