#!/usr/bin/env python3
"""Fixed judge extraction for Australian cases"""

import re
from collections import defaultdict
from load_real_aussie_corpus import corpus

class ImprovedJudgeAnalyzer:
    def __init__(self):
        # Initialize without loading corpus (will be loaded externally)
        self.judge_data = defaultdict(lambda: {
            'cases': [],
            'outcomes': defaultdict(int),
            'case_types': defaultdict(int),
            'total_cases': 0
        })
        self.analyzed = False
        
    def extract_judge_name(self, text):
        """Better extraction of judge names"""
        # More specific patterns for Australian courts
        patterns = [
            # "Coram: Smith J" or "Coram: Smith CJ"
            r'Coram:\s*([A-Z][a-z]+)\s+(?:CJ|J|JA|P|DP|JJ)\b',
            # "Before: Justice Smith" or "Before: Judge Smith"
            r'Before:\s*(?:Justice|Judge|The Hon(?:ourable)?\.?)\s+([A-Z][a-z]+)',
            # "Smith J:" at start of line
            r'^([A-Z][a-z]+)\s+(?:CJ|J|JA|P|DP|JJ):\s',
            # "The Honourable Justice Smith"
            r'The\s+Hon(?:ourable)?\.?\s+(?:Justice|Judge)\s+([A-Z][a-z]+)',
            # "SMITH J" in all caps
            r'\b([A-Z]{3,})\s+J\b(?!\w)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                # Return the first valid match
                for match in matches:
                    # Filter out common false positives
                    if match.upper() not in ['THE', 'OF', 'IN', 'TO', 'AND', 'OR', 'FOR', 'BY', 
                                           'WITH', 'FROM', 'BORDER', 'APPLICANT', 'RESPONDENT',
                                           'AUSTRALIA', 'GROUP', 'SERVICES', 'PTY', 'LTD', 'LIMITED']:
                        return match.upper()
        
        return None
    
    def analyze_all_judges(self):
        """Analyze all judges in the corpus"""
        if self.analyzed:
            return
            
        for case in corpus.cases:
            judge_name = self.extract_judge_name(case['text'])
            
            if judge_name:
                self.judge_data[judge_name]['cases'].append(case['citation'])
                self.judge_data[judge_name]['outcomes'][case['outcome']] += 1
                self.judge_data[judge_name]['total_cases'] += 1
                
                # Determine case type
                case_type = self._determine_case_type(case['text'])
                self.judge_data[judge_name]['case_types'][case_type] += 1
        
        self.analyzed = True
    
    def _determine_case_type(self, text):
        """Determine case type from text"""
        text_lower = text.lower()
        
        case_types = {
            'negligence': ['negligence', 'injury', 'accident'],
            'contract': ['contract', 'breach', 'agreement'],
            'employment': ['employment', 'dismissal', 'workplace'],
            'property': ['property', 'land', 'real estate'],
            'immigration': ['immigration', 'visa', 'refugee'],
            'criminal': ['criminal', 'offence', 'prosecution']
        }
        
        for case_type, keywords in case_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return case_type
        
        return 'other'

if __name__ == "__main__":
    # Test the improved extraction
    corpus.load_corpus()
    analyzer = ImprovedJudgeAnalyzer()
    
    print("Testing improved judge extraction on first 10 cases:")
    print("=" * 60)
    
    for i, case in enumerate(corpus.cases[:10]):
        judge = analyzer.extract_judge_name(case['text'])
        if judge:
            print(f"Case {i+1}: {case['citation'][:50]}")
            print(f"Judge found: {judge}")
            print("---")
