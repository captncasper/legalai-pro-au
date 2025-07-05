#!/usr/bin/env python3
"""Unit tests for the Australian Legal Corpus"""

from load_real_aussie_corpus import corpus

class CorpusUnitTests:
    """Test corpus functionality"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        
    def test_corpus_loading(self):
        """Test that corpus loads correctly"""
        corpus.load_corpus()
        
        # Test expected counts from your output
        expected_cases = 254
        expected_precedents = 307
        expected_judges = 4
        
        if len(corpus.cases) == expected_cases:
            self._pass(f"Loaded {expected_cases} cases")
        else:
            self._fail(f"Case count", f"Expected {expected_cases}, got {len(corpus.cases)}")
        
        if len(corpus.precedent_network) == expected_precedents:
            self._pass(f"Loaded {expected_precedents} precedent relationships")
        else:
            self._fail("Precedent count", f"Expected {expected_precedents}, got {len(corpus.precedent_network)}")
        
        if len(corpus.judge_patterns) >= expected_judges:
            self._pass(f"Loaded patterns for {len(corpus.judge_patterns)} judges")
        else:
            self._fail("Judge patterns", f"Expected at least {expected_judges}, got {len(corpus.judge_patterns)}")
    
    def test_outcome_distribution(self):
        """Test outcome distribution matches expected"""
        dist = corpus.get_outcome_distribution()
        
        # Expected from your output
        expected = {
            'applicant_lost': 163,
            'settled': 47,
            'applicant_won': 44
        }
        
        for outcome, count in expected.items():
            if dist.get(outcome) == count:
                self._pass(f"Outcome '{outcome}': {count} cases")
            else:
                self._fail(f"Outcome '{outcome}'", f"Expected {count}, got {dist.get(outcome, 0)}")
    
    def test_search_functionality(self):
        """Test search with known results"""
        # Test 1: Search for 'negligence' - should find 1 case
        results = corpus.search_cases("negligence")
        
        if len(results) == 1 and results[0]['citation'] == "Elysee v Ngo [2018] NSWDC 137":
            self._pass("Search 'negligence' found correct case")
        else:
            self._fail("Search 'negligence'", f"Expected 1 specific case, got {len(results)}")
        
        # Test 2: Search for court abbreviations
        results = corpus.search_cases("NSWDC")
        if results:
            self._pass(f"Search 'NSWDC' found {len(results)} cases")
        else:
            self._fail("Search 'NSWDC'", "No results found")
        
        # Test 3: Search for year
        results = corpus.search_cases("2018")
        if results:
            self._pass(f"Search '2018' found {len(results)} cases")
        else:
            self._fail("Search '2018'", "No results found")
    
    def test_case_structure(self):
        """Test that cases have expected structure"""
        if not corpus.cases:
            self._fail("Case structure", "No cases loaded")
            return
        
        # Check first case has required fields
        first_case = corpus.cases[0]
        required_fields = ['citation', 'outcome', 'text', 'case_name', 'year', 'court']
        
        for field in required_fields:
            if field in first_case:
                self._pass(f"Case has field '{field}'")
            else:
                self._fail(f"Case field '{field}'", "Missing")
    
    def test_specific_cases(self):
        """Test specific known cases"""
        # Test getting case by citation
        test_citations = [
            "Hardie v North Sydney City Council [2006] NSWLEC 45",
            "Elysee v Ngo [2018] NSWDC 137"
        ]
        
        for citation in test_citations:
            case = corpus.get_case_by_citation(citation)
            if case:
                self._pass(f"Found case: {citation[:50]}")
            else:
                self._fail(f"Get case by citation", f"Could not find {citation}")
    
    def test_precedent_network(self):
        """Test precedent relationships"""
        if corpus.precedent_network:
            # Check structure of first precedent relationship
            first_rel = corpus.precedent_network[0]
            
            if all(key in first_rel for key in ['citing', 'cited', 'strength']):
                self._pass("Precedent relationships have correct structure")
            else:
                self._fail("Precedent structure", "Missing required fields")
            
            # Test getting precedents for a case
            if corpus.precedent_network:
                test_case = corpus.precedent_network[0]['citing']
                rels = corpus.get_precedent_network_for_case(test_case)
                if rels:
                    self._pass(f"Found {len(rels)} precedent relationships")
                else:
                    self._fail("Precedent lookup", "No relationships found")
    
    def _pass(self, test_name: str):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def _fail(self, test_name: str, error: str):
        self.failed += 1
        print(f"❌ {test_name}: {error}")
    
    def run_all(self):
        """Run all unit tests"""
        print("\n🧪 Running Corpus Unit Tests")
        print("=" * 60)
        
        self.test_corpus_loading()
        self.test_outcome_distribution()
        self.test_search_functionality()
        self.test_case_structure()
        self.test_specific_cases()
        self.test_precedent_network()
        
        print("\n" + "=" * 60)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        
        total = self.passed + self.failed
        if total > 0:
            print(f"📊 Success Rate: {(self.passed/total*100):.1f}%")
        
        return self.failed == 0

if __name__ == "__main__":
    tester = CorpusUnitTests()
    success = tester.run_all()
    exit(0 if success else 1)
