#!/usr/bin/env python3
"""Simple integration tests using only standard library"""

import urllib.request
import urllib.parse
import json
import time
from typing import Dict, Any

from load_real_aussie_corpus import corpus

class SimpleAPITests:
    """Test API using urllib (no external dependencies)"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.passed = 0
        self.failed = 0
        
    def setup(self):
        """Load corpus"""
        print("\n🧪 Running Simple API Tests with Real Data")
        print("=" * 60)
        corpus.load_corpus()
        
    def make_request(self, endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> tuple:
        """Make HTTP request using urllib"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "POST" and data:
                json_data = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(url, data=json_data, method=method)
                req.add_header('Content-Type', 'application/json')
            else:
                req = urllib.request.Request(url, method=method)
            
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status, json.loads(response.read().decode('utf-8'))
        except Exception as e:
            return None, str(e)
    
    def test_health(self):
        """Test health endpoint"""
        status, data = self.make_request("/health")
        
        if status == 200:
            self._pass("Health check")
        else:
            self._fail("Health check", f"Status: {status}, Error: {data}")
    
    def test_search_real_cases(self):
        """Test search with real case data"""
        # Test with real search terms from your corpus
        test_queries = [
            "negligence",  # Found 1 case in corpus
            "Elysee v Ngo",  # Actual case citation
            "applicant",  # Common term
            "NSWDC",  # Court abbreviation
            "2018",  # Year
        ]
        
        for query in test_queries:
            status, data = self.make_request(
                "/api/v1/search/cases",
                "POST",
                {"query": query, "jurisdiction": "all"}
            )
            
            if status == 200:
                # Check if we got results
                if isinstance(data, dict) and ('results' in data or 'cases' in data):
                    results = data.get('results', data.get('cases', []))
                    self._pass(f"Search '{query}' - Found {len(results)} results")
                else:
                    self._pass(f"Search '{query}' - Response received")
            else:
                self._fail(f"Search '{query}'", f"Status: {status}")
    
    def test_case_analysis(self):
        """Test case analysis with real cases"""
        # Use actual cases from corpus
        real_cases = [
            {
                "case_name": "Elysee v Ngo",
                "citation": "Elysee v Ngo [2018] NSWDC 137",
                "jurisdiction": "nsw",
                "description": "negligence case"
            },
            {
                "case_name": "Hardie v North Sydney City Council",
                "citation": "Hardie v North Sydney City Council [2006] NSWLEC 45",
                "jurisdiction": "nsw",
                "description": "development application"
            }
        ]
        
        for case_data in real_cases[:1]:  # Test first case
            status, data = self.make_request(
                "/api/v1/analysis/quantum-supreme",
                "POST",
                case_data
            )
            
            if status == 200:
                self._pass(f"Analysis of {case_data['case_name']}")
            else:
                self._fail(f"Analysis of {case_data['case_name']}", f"Status: {status}")
    
    def test_statistics(self):
        """Test statistics endpoint"""
        status, data = self.make_request("/api/v1/admin/stats")
        
        if status == 200:
            if isinstance(data, dict):
                # Compare with actual corpus stats
                if 'corpus_size' in data:
                    print(f"   API reports: {data['corpus_size']} cases")
                    print(f"   Actual corpus: {len(corpus.cases)} cases")
                
                if 'outcome_distribution' in data:
                    print(f"   Outcome distribution: {data['outcome_distribution']}")
                
                self._pass("Statistics endpoint")
            else:
                self._pass("Statistics endpoint (basic response)")
        else:
            self._fail("Statistics endpoint", f"Status: {status}")
    
    def test_performance(self):
        """Simple performance test"""
        print("\n⚡ Performance Test:")
        
        times = []
        for i in range(5):
            start = time.time()
            status, _ = self.make_request(
                "/api/v1/search/cases",
                "POST",
                {"query": "contract", "jurisdiction": "all"}
            )
            if status == 200:
                elapsed = time.time() - start
                times.append(elapsed)
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"   Average response time: {avg_time:.3f}s")
            if avg_time < 1.0:
                self._pass("Performance test")
            else:
                self._fail("Performance test", f"Avg time {avg_time:.3f}s > 1s")
    
    def _pass(self, test_name: str):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def _fail(self, test_name: str, error: str):
        self.failed += 1
        print(f"❌ {test_name}: {error}")
    
    def run_all(self):
        """Run all tests"""
        self.setup()
        
        self.test_health()
        self.test_search_real_cases()
        self.test_case_analysis()
        self.test_statistics()
        self.test_performance()
        
        print("\n" + "=" * 60)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%")
        
        return self.failed == 0

if __name__ == "__main__":
    tester = SimpleAPITests()
    success = tester.run_all()
    exit(0 if success else 1)
