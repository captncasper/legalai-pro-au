import numpy as np
#!/usr/bin/env python3
"""Comprehensive test suite for MEGA Legal AI API"""

import requests
import json
import asyncio
import websocket
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

class MegaAPITester:
    def __init__(self):
        self.base_url = BASE_URL
        self.results = {"passed": 0, "failed": 0, "tests": []}
    
    def test_endpoint(self, name, method, endpoint, data=None, expected_status=200):
        """Test a single endpoint"""
        print(f"\n🧪 Testing: {name}")
        print(f"   Endpoint: {method} {endpoint}")
        
        try:
            if method == "GET":
                response = requests.get(f"{self.base_url}{endpoint}")
            elif method == "POST":
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
            
            success = response.status_code == expected_status
            
            if success:
                print(f"   ✅ Status: {response.status_code}")
                self.results["passed"] += 1
                
                # Print sample response
                try:
                    response_data = response.json()
                    if "results" in response_data:
                        print(f"   📊 Results preview: {json.dumps(response_data['results'], indent=2)[:200]}...")
                    elif "success" in response_data:
                        print(f"   📊 Success: {response_data['success']}")
                except:
                    pass
            else:
                print(f"   ❌ Status: {response.status_code} (expected {expected_status})")
                print(f"   Error: {response.text[:200]}")
                self.results["failed"] += 1
            
            self.results["tests"].append({
                "name": name,
                "endpoint": endpoint,
                "status": "passed" if success else "failed",
                "response_time": response.elapsed.total_seconds()
            })
            
            return success, response
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            self.results["failed"] += 1
            return False, None
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 MEGA Legal AI API - Comprehensive Test Suite")
        print("=" * 60)
        
        # 1. General endpoints
        print("\n📍 TESTING GENERAL ENDPOINTS")
        self.test_endpoint("Root", "GET", "/")
        self.test_endpoint("Health Check", "GET", "/health")
        
        # 2. Quantum Analysis
        print("\n🌌 TESTING QUANTUM ANALYSIS")
        quantum_data = {
            "case_type": "employment",
            "description": "Wrongful termination with discrimination",
            "jurisdiction": "NSW",
            "arguments": [
                "No written warnings provided",
                "Termination after complaint",
                "Pattern of discrimination",
                "Excellent performance reviews"
            ],
            "precedents": ["Smith v ABC Corp", "Jones v XYZ Ltd"],
            "evidence_strength": 85
        }
        self.test_endpoint("Quantum Analysis", "POST", "/api/v1/analysis/quantum", quantum_data)
        
        # 3. Monte Carlo Simulation
        print("\n🎲 TESTING MONTE CARLO SIMULATION")
        simulation_data = {
            "case_data": {
                "case_type": "employment",
                "strength_score": 75,
                "precedent_support": 80,
                "jurisdiction": "NSW"
            },
            "prediction_type": "outcome",
            "num_simulations": 5000
        }
        self.test_endpoint("Monte Carlo Simulation", "POST", "/api/v1/prediction/simulate", simulation_data)
        
        # 4. Emotion Analysis
        print("\n😊 TESTING EMOTION ANALYSIS")
        emotion_data = {
            "text": "I am extremely frustrated and disappointed by the company's discriminatory actions. The unfair treatment has caused significant emotional distress.",
            "context": "victim_statement"
        }
        self.test_endpoint("Emotion Analysis", "POST", "/api/v1/analysis/emotion", emotion_data)
        
        # 5. Pattern Recognition
        print("\n🔍 TESTING PATTERN RECOGNITION")
        pattern_data = {
            "case_description": "Employee terminated after raising safety concerns. Company has history of retaliating against whistleblowers. Similar cases in the industry.",
            "pattern_type": "all",
            "depth": 3
        }
        self.test_endpoint("Pattern Recognition", "POST", "/api/v1/analysis/pattern", pattern_data)
        
        # 6. Risk Assessment
        print("\n⚠️ TESTING RISK ASSESSMENT")
        risk_data = {
            "case_data": {
                "case_type": "commercial_litigation",
                "claim_amount": 2500000,
                "opponent": "Large Corporation",
                "jurisdiction": "Federal"
            },
            "risk_factors": ["novel_legal_theory", "media_attention", "precedent_setting"],
            "timeline": "18_months"
        }
        self.test_endpoint("Risk Assessment", "POST", "/api/v1/analysis/risk", risk_data)
        
        # 7. Document Generation
        print("\n📄 TESTING DOCUMENT GENERATION")
        doc_data = {
            "document_type": "contract",
            "context": {
                "parties": ["Tech Innovations Pty Ltd", "John Smith"],
                "purpose": "software development services",
                "compensation": "$150,000 AUD",
                "start_date": "July 1, 2024",
                "end_date": "December 31, 2024"
            },
            "style": "formal",
            "length": "standard"
        }
        self.test_endpoint("Document Generation - Contract", "POST", "/api/v1/generate/document", doc_data)
        
        # Test Brief Generation
        brief_data = {
            "document_type": "brief",
            "context": {
                "case_name": "Smith v Tech Corp",
                "court": "Federal Court of Australia",
                "statement": "This case involves wrongful termination and discrimination",
                "facts": "1. Employee worked for 5 years\n2. Excellent reviews\n3. Terminated after complaint",
                "argument_1_title": "Wrongful Termination",
                "argument_1": "The termination violated Fair Work Act provisions..."
            },
            "style": "formal"
        }
        self.test_endpoint("Document Generation - Brief", "POST", "/api/v1/generate/document", brief_data)
        
        # 8. Settlement Calculation
        print("\n💰 TESTING SETTLEMENT CALCULATOR")
        settlement_data = {
            "case_type": "personal_injury",
            "claim_amount": 500000,
            "injury_severity": "severe",
            "liability_admission": True,
            "negotiation_stage": "mediation"
        }
        self.test_endpoint("Settlement Calculation", "POST", "/api/v1/calculate/settlement", settlement_data)
        
        # 9. Voice Commands
        print("\n🎤 TESTING VOICE COMMANDS")
        voice_data = {
            "command": "Analyze the liability factors in this employment case",
            "context": {"case_type": "employment", "jurisdiction": "NSW"}
        }
        self.test_endpoint("Voice Command", "POST", "/api/v1/voice/command", voice_data)
        
        # 10. Collaboration
        print("\n👥 TESTING COLLABORATION")
        collab_data = {
            "case_id": "CASE-2024-001",
            "user_id": "lawyer123",
            "action": "create"
        }
        self.test_endpoint("Create Collaboration", "POST", "/api/v1/collaborate/create", collab_data)
        
        # 11. Search
        print("\n🔎 TESTING SEARCH")
        search_data = {
            "query": "wrongful termination discrimination damages NSW",
            "search_type": "hybrid",
            "filters": {"jurisdiction": "NSW", "year_from": 2020},
            "limit": 10
        }
        self.test_endpoint("Case Search", "POST", "/api/v1/search/cases", search_data)
        
        # 12. Admin Endpoints
        print("\n🔧 TESTING ADMIN ENDPOINTS")
        self.test_endpoint("System Stats", "GET", "/api/v1/admin/stats")
        self.test_endpoint("Clear Cache", "POST", "/api/v1/admin/cache/clear")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"📈 Success Rate: {(self.results['passed'] / (self.results['passed'] + self.results['failed']) * 100):.1f}%")
        
        # Response time analysis
        response_times = [t['response_time'] for t in self.results['tests'] if 'response_time' in t]
        if response_times:
            print(f"\n⏱️  Performance Metrics:")
            print(f"   Average Response Time: {sum(response_times)/len(response_times)*1000:.0f}ms")
            print(f"   Fastest: {min(response_times)*1000:.0f}ms")
            print(f"   Slowest: {max(response_times)*1000:.0f}ms")
        
        print("\n" + "="*60)

def test_websocket():
    """Test WebSocket connection"""
    print("\n🔌 TESTING WEBSOCKET")
    try:
        ws = websocket.WebSocket()
        ws.connect("ws://localhost:8000/ws/assistant")
        
        # Test connection
        response = json.loads(ws.recv())
        print(f"   ✅ Connected: {response['message']}")
        
        # Test chat
        ws.send(json.dumps({
            "type": "chat",
            "query": "What is wrongful termination?"
        }))
        response = json.loads(ws.recv())
        print(f"   ✅ Chat response received")
        
        # Test analysis
        ws.send(json.dumps({
            "type": "analyze",
            "analysis_type": "quantum",
            "params": {
                "case_type": "employment",
                "arguments": ["test"]
            }
        }))
        response = json.loads(ws.recv())
        print(f"   ✅ Analysis response received")
        
        ws.close()
        print(f"   ✅ WebSocket test completed")
        
    except Exception as e:
        print(f"   ❌ WebSocket error: {e}")

if __name__ == "__main__":
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Run main tests
    tester = MegaAPITester()
    tester.run_all_tests()
    
    # Test WebSocket
    test_websocket()
    
    print("\n✨ All tests completed!")
