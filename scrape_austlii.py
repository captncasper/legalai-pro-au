#!/usr/bin/env python3
"""Scrape more cases from AustLII"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime

class AustLIIScraper:
    def __init__(self):
        self.base_url = "http://www.austlii.edu.au"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Legal Research Bot 1.0 (Educational Purpose)'
        })
        
    def search_cases(self, query, jurisdiction="nsw", year=2024, max_results=50):
        """Search for cases on AustLII"""
        print(f"🔍 Searching AustLII for: {query}")
        
        # Build search URL
        search_params = {
            'method': 'boolean',
            'query': query,
            'meta': f'/au/cases/{jurisdiction}/{year}',
            'results': max_results
        }
        
        search_url = f"{self.base_url}/cgi-bin/sinosrch.cgi"
        
        try:
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find case links
            cases = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/au/cases/' in href and href.endswith('.html'):
                    case_url = f"{self.base_url}{href}"
                    case_title = link.text.strip()
                    
                    if case_title:
                        cases.append({
                            'title': case_title,
                            'url': case_url,
                            'jurisdiction': jurisdiction,
                            'year': year
                        })
            
            print(f"✅ Found {len(cases)} cases")
            return cases[:max_results]
            
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []
    
    def extract_case_details(self, case_url):
        """Extract details from a case page"""
        try:
            response = self.session.get(case_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract case details
            details = {
                'url': case_url,
                'full_text': soup.get_text()[:5000],  # First 5000 chars
                'date_scraped': datetime.now().isoformat()
            }
            
            # Try to extract specific fields
            title_tag = soup.find('title')
            if title_tag:
                details['title'] = title_tag.text.strip()
            
            # Look for citation
            citation_patterns = [
                r'\[\d{4}\]\s+\w+\s+\d+',
                r'\(\d{4}\)\s+\d+\s+\w+\s+\d+'
            ]
            
            for pattern in citation_patterns:
                import re
                match = re.search(pattern, details['full_text'])
                if match:
                    details['citation'] = match.group()
                    break
            
            return details
            
        except Exception as e:
            print(f"❌ Error extracting case: {e}")
            return None
    
    def scrape_recent_cases(self, jurisdictions=['nsw', 'vic', 'qld'], 
                           queries=['negligence', 'contract', 'employment'],
                           max_per_query=10):
        """Scrape recent cases from multiple jurisdictions"""
        all_cases = []
        
        for jurisdiction in jurisdictions:
            for query in queries:
                print(f"\n📍 Searching {jurisdiction.upper()} for '{query}'...")
                
                cases = self.search_cases(query, jurisdiction, 2024, max_per_query)
                
                for case in cases[:5]:  # Limit to avoid overwhelming
                    print(f"  Extracting: {case['title'][:50]}...")
                    details = self.extract_case_details(case['url'])
                    
                    if details:
                        details.update(case)
                        all_cases.append(details)
                    
                    time.sleep(1)  # Be respectful
        
        # Save results
        with open('austlii_cases_new.json', 'w') as f:
            json.dump(all_cases, f, indent=2)
        
        print(f"\n✅ Scraped {len(all_cases)} new cases")
        print("💾 Saved to: austlii_cases_new.json")
        
        return all_cases

if __name__ == "__main__":
    scraper = AustLIIScraper()
    
    # Example: Get recent negligence cases from NSW
    cases = scraper.search_cases("negligence damages", "nsw", 2024, 10)
    
    if cases:
        print("\n📋 Sample cases found:")
        for case in cases[:5]:
            print(f"  - {case['title']}")
    
    # Uncomment to actually scrape (be respectful of AustLII)
    # scraper.scrape_recent_cases()
