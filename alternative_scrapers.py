#!/usr/bin/env python3
"""
Alternative scrapers for Australian legal sources
Includes Federal Court, High Court, and Commonwealth Courts
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import feedparser
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlternativeScrapers:
    """Alternative sources for Australian legal cases"""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Legal Research Bot 1.0 (Educational/Research Purpose)'
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def scrape_federal_court(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Scrape Federal Court of Australia decisions
        Uses RSS feeds and search functionality
        """
        logger.info(f"🏛️ Searching Federal Court for: {query}")
        cases = []
        
        try:
            # Federal Court RSS feed for recent decisions
            rss_url = "https://www.fedcourt.gov.au/rss/judgments-rss"
            
            async with self.session.get(rss_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:max_results]:
                        # Check if query matches
                        if query.lower() in entry.title.lower() or query.lower() in entry.summary.lower():
                            case = {
                                'title': entry.title,
                                'url': entry.link,
                                'date': entry.published,
                                'summary': BeautifulSoup(entry.summary, 'html.parser').get_text()[:500],
                                'court': 'FCA',
                                'source': 'federal_court_rss'
                            }
                            
                            # Extract citation from title
                            citation_match = re.search(r'\[(\d{4})\]\s*(FCA|FCAFC)\s*(\d+)', entry.title)
                            if citation_match:
                                case['citation'] = citation_match.group(0)
                                case['year'] = int(citation_match.group(1))
                                case['number'] = citation_match.group(3)
                            
                            cases.append(case)
            
            # Also try the search page
            search_url = "https://search2.fedcourt.gov.au/s/search.html"
            search_params = {
                'collection': 'fedcourt-judgments',
                'query': query,
                'sort': 'date'
            }
            
            async with self.session.get(search_url, params=search_params) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find search results
                    results = soup.find_all('div', class_='search-result')
                    
                    for result in results[:max_results - len(cases)]:
                        title_elem = result.find('h3', class_='title')
                        if title_elem:
                            case = {
                                'title': title_elem.text.strip(),
                                'url': title_elem.find('a')['href'] if title_elem.find('a') else '',
                                'summary': result.find('p', class_='summary').text.strip() if result.find('p', class_='summary') else '',
                                'court': 'FCA',
                                'source': 'federal_court_search'
                            }
                            
                            # Extract citation
                            citation_match = re.search(r'\[(\d{4})\]\s*(FCA|FCAFC)\s*(\d+)', case['title'])
                            if citation_match:
                                case['citation'] = citation_match.group(0)
                                case['year'] = int(citation_match.group(1))
                            
                            cases.append(case)
                            
        except Exception as e:
            logger.error(f"Federal Court scraping error: {e}")
        
        logger.info(f"Found {len(cases)} Federal Court cases")
        return cases
    
    async def scrape_high_court(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Scrape High Court of Australia decisions
        """
        logger.info(f"⚖️ Searching High Court for: {query}")
        cases = []
        
        try:
            # High Court recent decisions page
            url = "https://www.hcourt.gov.au/cases/recent-decisions"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find case entries
                    case_entries = soup.find_all('div', class_='views-row')
                    
                    for entry in case_entries[:max_results]:
                        title_elem = entry.find('h3') or entry.find('h4')
                        if title_elem and query.lower() in title_elem.text.lower():
                            case = {
                                'title': title_elem.text.strip(),
                                'court': 'HCA',
                                'source': 'high_court'
                            }
                            
                            # Extract link
                            link_elem = entry.find('a')
                            if link_elem:
                                case['url'] = f"https://www.hcourt.gov.au{link_elem['href']}"
                            
                            # Extract date
                            date_elem = entry.find('span', class_='date-display-single')
                            if date_elem:
                                case['date'] = date_elem.text.strip()
                            
                            # Extract citation
                            citation_match = re.search(r'\[(\d{4})\]\s*HCA\s*(\d+)', case['title'])
                            if citation_match:
                                case['citation'] = citation_match.group(0)
                                case['year'] = int(citation_match.group(1))
                                case['number'] = citation_match.group(2)
                            
                            # Extract summary
                            summary_elem = entry.find('div', class_='field-content')
                            if summary_elem:
                                case['summary'] = summary_elem.text.strip()[:500]
                            
                            cases.append(case)
                            
        except Exception as e:
            logger.error(f"High Court scraping error: {e}")
        
        logger.info(f"Found {len(cases)} High Court cases")
        return cases
    
    async def scrape_comcourts(self, query: str, jurisdiction: str = 'all', max_results: int = 20) -> List[Dict]:
        """
        Scrape Commonwealth Courts Portal
        """
        logger.info(f"🏛️ Searching Commonwealth Courts for: {query}")
        cases = []
        
        try:
            # Build search URL based on jurisdiction
            base_url = "https://www.comcourts.gov.au"
            
            jurisdiction_map = {
                'federal': '/judgments/search?court=fca',
                'family': '/judgments/search?court=fcfca',
                'circuit': '/judgments/search?court=fcca'
            }
            
            if jurisdiction in jurisdiction_map:
                search_path = jurisdiction_map[jurisdiction]
            else:
                search_path = '/judgments/search'
            
            search_url = f"{base_url}{search_path}"
            
            # Search parameters
            params = {
                'search': query,
                'sort': 'date_desc',
                'page_size': max_results
            }
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find judgment entries
                    judgment_list = soup.find('div', class_='judgment-list')
                    if judgment_list:
                        entries = judgment_list.find_all('div', class_='judgment-item')
                        
                        for entry in entries:
                            case = {
                                'source': 'commonwealth_courts'
                            }
                            
                            # Extract title and citation
                            title_elem = entry.find('h2', class_='judgment-title')
                            if title_elem:
                                case['title'] = title_elem.text.strip()
                                
                                # Extract citation from title
                                citation_match = re.search(r'\[(\d{4})\]\s*(\w+)\s*(\d+)', case['title'])
                                if citation_match:
                                    case['citation'] = citation_match.group(0)
                                    case['year'] = int(citation_match.group(1))
                                    case['court'] = citation_match.group(2)
                                    case['number'] = citation_match.group(3)
                            
                            # Extract URL
                            link_elem = title_elem.find('a') if title_elem else None
                            if link_elem:
                                case['url'] = f"{base_url}{link_elem['href']}"
                            
                            # Extract date
                            date_elem = entry.find('span', class_='judgment-date')
                            if date_elem:
                                case['date'] = date_elem.text.strip()
                            
                            # Extract summary/catchwords
                            catchwords_elem = entry.find('div', class_='catchwords')
                            if catchwords_elem:
                                case['catchwords'] = catchwords_elem.text.strip()
                            
                            cases.append(case)
                            
        except Exception as e:
            logger.error(f"Commonwealth Courts scraping error: {e}")
        
        logger.info(f"Found {len(cases)} Commonwealth Courts cases")
        return cases
    
    async def scrape_nsw_caselaw(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Scrape NSW Caselaw (replacement for AustLII NSW)
        """
        logger.info(f"🏛️ Searching NSW Caselaw for: {query}")
        cases = []
        
        try:
            # NSW Caselaw search
            search_url = "https://www.caselaw.nsw.gov.au/search/advanced"
            
            # This would need proper form submission
            # For now, using the browse recent decisions
            browse_url = "https://www.caselaw.nsw.gov.au/browse/recent"
            
            async with self.session.get(browse_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find case entries
                    case_list = soup.find_all('tr', class_='decision-row')
                    
                    for row in case_list[:max_results]:
                        if query.lower() in row.text.lower():
                            case = {
                                'court': 'NSW',
                                'source': 'nsw_caselaw'
                            }
                            
                            # Extract citation
                            citation_elem = row.find('td', class_='citation')
                            if citation_elem:
                                case['citation'] = citation_elem.text.strip()
                                
                                # Extract year
                                year_match = re.search(r'\[(\d{4})\]', case['citation'])
                                if year_match:
                                    case['year'] = int(year_match.group(1))
                            
                            # Extract title
                            title_elem = row.find('td', class_='title')
                            if title_elem:
                                case['title'] = title_elem.text.strip()
                                case['case_name'] = case['title']
                            
                            # Extract URL
                            link_elem = row.find('a')
                            if link_elem:
                                case['url'] = f"https://www.caselaw.nsw.gov.au{link_elem['href']}"
                            
                            # Extract date
                            date_elem = row.find('td', class_='date')
                            if date_elem:
                                case['date'] = date_elem.text.strip()
                            
                            cases.append(case)
                            
        except Exception as e:
            logger.error(f"NSW Caselaw scraping error: {e}")
        
        logger.info(f"Found {len(cases)} NSW Caselaw cases")
        return cases
    
    async def scrape_all_sources(self, query: str, max_per_source: int = 10) -> List[Dict]:
        """
        Search all available sources
        """
        logger.info(f"🔍 Searching all sources for: {query}")
        all_cases = []
        
        # Run all scrapers concurrently
        tasks = [
            self.scrape_federal_court(query, max_per_source),
            self.scrape_high_court(query, max_per_source),
            self.scrape_comcourts(query, 'all', max_per_source),
            self.scrape_nsw_caselaw(query, max_per_source)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, list):
                all_cases.extend(result)
            else:
                logger.error(f"Scraper {i} failed: {result}")
        
        # Remove duplicates based on citation
        seen_citations = set()
        unique_cases = []
        
        for case in all_cases:
            citation = case.get('citation', case.get('title', ''))
            if citation and citation not in seen_citations:
                seen_citations.add(citation)
                unique_cases.append(case)
        
        logger.info(f"✅ Total unique cases found: {len(unique_cases)}")
        return unique_cases

# Example usage
async def test_alternative_scrapers():
    """Test the alternative scrapers"""
    async with AlternativeScrapers() as scrapers:
        # Test Federal Court
        print("\n🏛️ Testing Federal Court scraper...")
        federal_cases = await scrapers.scrape_federal_court("artificial intelligence", 5)
        for case in federal_cases[:2]:
            print(f"  - {case.get('citation', case.get('title', 'Unknown'))}")
        
        # Test High Court
        print("\n⚖️ Testing High Court scraper...")
        high_cases = await scrapers.scrape_high_court("negligence", 5)
        for case in high_cases[:2]:
            print(f"  - {case.get('citation', case.get('title', 'Unknown'))}")
        
        # Test all sources
        print("\n🔍 Testing all sources...")
        all_cases = await scrapers.scrape_all_sources("contract breach", 5)
        print(f"Found {len(all_cases)} total cases across all sources")

if __name__ == "__main__":
    asyncio.run(test_alternative_scrapers())
