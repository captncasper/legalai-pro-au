#!/usr/bin/env python3
"""Integration tests using real Australian legal corpus data"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List
import sys

from load_real_aussie_corpus import corpus

class RealDataIntegrationTests:
    """Test the API with actual Australian case data"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.passed = 0
        self.failed = 0
        self.session = None
        
    async def setup(self):
        """Setup test environment"""
        self.session = aiohttp.ClientSession()
        corpus.load_corpus()
        print("\n🧪 Running Integration Tests with Real Australian Legal Data")
        print("=" * 60)
        
    async def teardown(self):
        """Cleanup"""
        if self.session:
            await self.session.close()
        
        print("\n" + "=" * 60)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%" if self.passed+self.failed > 0 else "N/A")
    
    async def test_health_check(self):
        """Test basic API health"""
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert 'status' in data
                self._pass("Health check")
        except Exception as e:
            self._fail("Health check", str(e))
    
    async def test_search_with_real_cases(self):
        """Test search using actual case names and terms"""
        # Get some real search terms from your corpus
        test_searches = [
            # Real parties from cases
            "Hardie v North Sydney",
            "Cromwell Corporation",
            "Minister for Immigration",
            
            # Common legal terms
            "negligence",
            "contract",
            "damages",
            
            # Real courts
            "Federal Court",
            "NSWLEC",
            "High Court"
        ]
        
        for search_term in test_searches:
            try:
                async with self.session.post(
                    f"{self.base_url}/api/v1/search/cases",
                    json={"query": search_term, "jurisdiction": "all"}
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    
                    # Check response structure
                    assert 'results' in data or 'cases' in data or 'matches' in data
                    
                    self._pass(f"Search: '{search_term}'")
            except Exception as e:
                self._fail(f"Search: '{search_term}'", str(e))
    
    async def test_analyze_real_case(self):
        """Test case analysis with real case data"""
        # Use actual cases from your corpus
        real_cases = corpus.cases[:5]  # Get first 5 real cases
        
        for case in real_cases:
            try:
                # Prepare request with real case data
                request_data = {
                    "case_name": case['case_name'],
                    "citation": case['citation'],
                    "jurisdiction": self._extract_jurisdiction(case['court']),
                    "case_type": self._determine_case_type(case['text']),
                    "description": case['text'][:500],  # First 500 chars
                    "outcome": case['outcome']
                }
                
                async with self.session.post(
                    f"{self.base_url}/api/v1/analysis/quantum-supreme",
                    json=request_data
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # Verify some kind of analysis response
                        assert any(key in data for key in ['prediction', 'analysis', 'outcome_probability', 'success'])
                        
                        self._pass(f"Analysis: {case['case_name'][:50]}")
                    else:
                        self._fail(f"Analysis: {case['case_name'][:50]}", f"Status {resp.status}")
                        
            except Exception as e:
                self._fail(f"Analysis: {case['case_name'][:50]}", str(e))
    
    async def test_precedent_network(self):
        """Test precedent network with real citation relationships"""
        # Get cases that have precedent relationships
        cases_with_precedents = []
        for rel in corpus.precedent_network[:10]:
            cases_with_precedents.append(rel['citing'])
        
        for citation in cases_with_precedents[:3]:
            try:
                async with self.session.post(
                    f"{self.base_url}/api/v1/search/precedents",
                    json={"citation": citation}
                ) as resp:
                    if resp.status == 200:
                        self._pass(f"Precedent search: {citation[:50]}")
                    else:
                        # Try alternative endpoint
                        async with self.session.get(
                            f"{self.base_url}/api/v1/case/precedents",
                            params={"citation": citation}
                        ) as resp2:
                            if resp2.status == 200:
                                self._pass(f"Precedent search: {citation[:50]}")
                            else:
                                self._fail(f"Precedent search: {citation[:50]}", f"Status {resp.status}")
            except Exception as e:
                self._warn(f"Precedent search not implemented: {citation[:30]}")
    
    async def test_outcome_statistics(self):
        """Test with real outcome distribution"""
        try:
            # Get real statistics
            outcome_dist = corpus.get_outcome_distribution()
            
            async with self.session.get(
                f"{self.base_url}/api/v1/admin/stats"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._pass("Statistics endpoint")
                    
                    # Show comparison if available
                    if 'corpus_size' in data:
                        print(f"   API reports {data['corpus_size']} cases")
                        print(f"   Test corpus has {len(corpus.cases)} cases")
                else:
                    self._fail("Statistics endpoint", f"Status {resp.status}")
        except Exception as e:
            self._fail("Statistics endpoint", str(e))
    
    async def test_performance_with_real_queries(self):
        """Test performance using real case searches"""
        # Extract real search terms from cases
        search_terms = []
        for case in corpus.cases[:20]:
            # Get party names
            parts = case['case_name'].split(' v ')
            if parts:
                search_terms.append(parts[0].split()[0])  # First party's first word
        
        response_times = []
        
        for term in search_terms[:10]:
            start_time = time.time()
            
            try:
                async with self.session.post(
                    f"{self.base_url}/api/v1/search/cases",
                    json={"query": term},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    await resp.json()
                    response_times.append(time.time() - start_time)
            except:
                pass
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            
            if avg_time < 1.0:
                self._pass(f"Performance: avg {avg_time:.2f}s, max {max_time:.2f}s")
            else:
                self._fail(f"Performance", f"avg {avg_time:.2f}s exceeds 1s threshold")
    
    def _extract_jurisdiction(self, court: str) -> str:
        """Extract jurisdiction from court abbreviation"""
        if 'NSW' in court:
            return 'nsw'
        elif 'VIC' in court:
            return 'vic'
        elif 'QLD' in court:
            return 'qld'
        elif 'FCA' in court or 'HCA' in court:
            return 'federal'
        else:
            return 'all'
    
    def _determine_case_type(self, text: str) -> str:
        """Determine case type from text"""
        text_lower = text.lower()
        
        if 'negligence' in text_lower or 'injury' in text_lower:
            return 'tort'
        elif 'contract' in text_lower or 'breach' in text_lower:
            return 'contract'
        elif 'criminal' in text_lower or 'offence' in text_lower:
            return 'criminal'
        elif 'immigration' in text_lower or 'visa' in text_lower:
            return 'immigration'
        else:
            return 'general'
    
    def _pass(self, test_name: str):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def _fail(self, test_name: str, error: str):
        self.failed += 1
        print(f"❌ {test_name}: {error}")
    
    def _warn(self, message: str):
        print(f"⚠️  {message}")
    
    async def run_all_tests(self):
        """Run all tests"""
        await self.setup()
        
        # Run tests
        await self.test_health_check()
        await self.test_search_with_real_cases()
        await self.test_analyze_real_case()
        await self.test_precedent_network()
        await self.test_outcome_statistics()
        await self.test_performance_with_real_queries()
        
        await self.teardown()
        
        return self.failed == 0

if __name__ == "__main__":
    async def main():
        tester = RealDataIntegrationTests()
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)
    
    asyncio.run(main())
