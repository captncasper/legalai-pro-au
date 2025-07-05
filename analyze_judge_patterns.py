#!/usr/bin/env python3
"""Analyze judge patterns from your corpus"""

import re
from collections import defaultdict
from load_real_aussie_corpus import corpus

class JudgeAnalyzer:
    def __init__(self):
        corpus.load_corpus()
        self.judge_data = defaultdict(lambda: {
            'cases': [],
            'outcomes': defaultdict(int),
            'case_types': defaultdict(int),
            'total_cases': 0
        })
        
    def extract_judge_name(self, text):
        """Extract judge name from case text"""
        patterns = [
            r'(?:Justice|Judge|Magistrate|Hon(?:ourable)?\.?)\s+([A-Z][a-zA-Z]+)',
            r'([A-Z][a-zA-Z]+)\s+(?:J|JA|CJ|P|DP|JJ)',
            r'Coram:\s*([A-Z][a-zA-Z]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    def analyze_judges(self):
        """Analyze judge patterns in corpus"""
        
        # Extract judge info from each case
        for case in corpus.cases:
            judge_name = self.extract_judge_name(case['text'])
            
            if judge_name:
                self.judge_data[judge_name]['cases'].append(case['citation'])
                self.judge_data[judge_name]['outcomes'][case['outcome']] += 1
                self.judge_data[judge_name]['total_cases'] += 1
                
                # Determine case type
                case_type = self._determine_case_type(case['text'])
                self.judge_data[judge_name]['case_types'][case_type] += 1
        
        # Analysis
        print("\n👨‍⚖️ Judge Pattern Analysis:")
        print(f"Judges identified: {len(self.judge_data)}")
        
        # Top judges by case count
        print("\n📊 Most active judges:")
        sorted_judges = sorted(self.judge_data.items(), 
                              key=lambda x: x[1]['total_cases'], 
                              reverse=True)
        
        for judge, data in sorted_judges[:10]:
            total = data['total_cases']
            win_rate = data['outcomes'].get('applicant_won', 0) / total * 100
            settle_rate = data['outcomes'].get('settled', 0) / total * 100
            
            print(f"\n  Judge {judge}:")
            print(f"    Total cases: {total}")
            print(f"    Applicant win rate: {win_rate:.1f}%")
            print(f"    Settlement rate: {settle_rate:.1f}%")
            print(f"    Primary case types: {', '.join(list(data['case_types'].keys())[:3])}")
        
        return dict(self.judge_data)
    
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
    analyzer = JudgeAnalyzer()
    analyzer.analyze_judges()
