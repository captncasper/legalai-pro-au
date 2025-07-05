#!/usr/bin/env python3
"""Extract settlement amounts from your existing corpus"""

import re
from load_real_aussie_corpus import corpus

class SettlementExtractor:
    def __init__(self):
        # Australian currency patterns
        self.money_patterns = [
            r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|mil|m|thousand|k))?',
            r'[\d,]+(?:\.\d{2})?\s*dollars',
            r'AUD\s*[\d,]+(?:\.\d{2})?',
            r'damages.*?of\s*\$[\d,]+',
            r'awarded.*?\$[\d,]+',
            r'settlement.*?\$[\d,]+',
            r'compensation.*?\$[\d,]+',
        ]
        
    def extract_amounts(self, text):
        """Extract monetary amounts from text"""
        amounts = []
        
        for pattern in self.money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_amount = self._parse_amount(match)
                if clean_amount > 0:
                    amounts.append(clean_amount)
        
        return amounts
    
    def _parse_amount(self, amount_str):
        """Parse amount string to float"""
        try:
            # Remove currency symbols and spaces
            clean = amount_str.replace('$', '').replace(',', '').replace('AUD', '').strip()
            
            # Handle multipliers
            multiplier = 1
            if 'million' in clean or ' m' in clean:
                multiplier = 1_000_000
                clean = re.sub(r'million|mil|m', '', clean, flags=re.IGNORECASE).strip()
            elif 'thousand' in clean or ' k' in clean:
                multiplier = 1_000
                clean = re.sub(r'thousand|k', '', clean, flags=re.IGNORECASE).strip()
            
            # Extract number
            number_match = re.search(r'[\d.]+', clean)
            if number_match:
                return float(number_match.group()) * multiplier
        except:
            pass
        
        return 0
    
    def analyze_corpus_settlements(self):
        """Analyze settlement amounts in corpus"""
        corpus.load_corpus()
        
        settlement_data = []
        cases_with_amounts = 0
        
        for case in corpus.cases:
            amounts = self.extract_amounts(case['text'])
            
            if amounts:
                cases_with_amounts += 1
                settlement_data.append({
                    'citation': case['citation'],
                    'outcome': case['outcome'],
                    'amounts': amounts,
                    'max_amount': max(amounts),
                    'year': case['year']
                })
        
        # Analysis
        print(f"\n💰 Settlement Amount Analysis:")
        print(f"Cases with monetary amounts: {cases_with_amounts}/{len(corpus.cases)}")
        
        if settlement_data:
            # By outcome
            outcomes = {}
            for data in settlement_data:
                outcome = data['outcome']
                if outcome not in outcomes:
                    outcomes[outcome] = []
                outcomes[outcome].append(data['max_amount'])
            
            print("\n📊 Average amounts by outcome:")
            for outcome, amounts in outcomes.items():
                avg = sum(amounts) / len(amounts)
                print(f"  {outcome}: ${avg:,.0f} (n={len(amounts)})")
            
            # Top settlements
            print("\n🏆 Top 5 settlements found:")
            sorted_data = sorted(settlement_data, key=lambda x: x['max_amount'], reverse=True)
            for data in sorted_data[:5]:
                print(f"  {data['citation']}: ${data['max_amount']:,.0f}")
        
        return settlement_data

if __name__ == "__main__":
    extractor = SettlementExtractor()
    extractor.analyze_corpus_settlements()
