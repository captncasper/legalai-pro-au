#!/usr/bin/env python3
"""
Intelligent Legal Scraper - Automatically fetches cases when needed
Scrapes from multiple sources including AustLII
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentLegalScraper:
    """Smart scraper that knows when and what to scrape"""
    
    def __init__(self):
        self.sources = {
            'austlii': {
                'base_url': 'http://www.austlii.edu.au',
                'search_url': 'http://www.austlii.edu.au/cgi-bin/sinosrch.cgi',
                'rate_limit': 1.0  # seconds between requests
            },
            'jade': {
                'base_url': 'https://jade.io/j/',  # Note: Requires subscription
                'rate_limit': 2.0
            }
        }
        
        self.headers = {
            'User-Agent': 'Legal Research Bot 1.0 (Educational/Research Purpose)'
        }
        
        self.cache_dir = Path("scraped_cases")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def smart_search(self, query: str, context: Dict = None) -> List[Dict]:
        """
        Intelligently search for cases based on query and context
        
        Args:
            query: Search query
            context: Additional context like jurisdiction, year, case type
        
        Returns:
            List of relevant cases
        """
        logger.info(f"🔍 Smart search for: {query}")
        
        # Determine what to search for
        search_params = self._analyze_search_needs(query, context)
        
        # Search across multiple sources
        results = []
        
        # Always try AustLII first (free)
        austlii_results = await self._search_austlii(
            query=search_params['query'],
            jurisdiction=search_params.get('jurisdiction', 'all'),
            year_from=search_params.get('year_from', 2020),
            max_results=search_params.get('max_results', 20)
        )
        results.extend(austlii_results)
        
        # If not enough results, expand search
        if len(results) < 10:
            logger.info("📈 Expanding search parameters...")
            # Try broader search
            broader_results = await self._search_austlii(
                query=self._broaden_query(query),
                jurisdiction='all',
                year_from=2015,
                max_results=20
            )
            results.extend(broader_results)
        
        # Deduplicate
        results = self._deduplicate_results(results)
        
        # Rank by relevance
        results = self._rank_results(results, query, context)
        
        logger.info(f"✅ Found {len(results)} relevant cases")
        return results
    
    def _analyze_search_needs(self, query: str, context: Dict = None) -> Dict:
        """Analyze what kind of search is needed"""
        params = {
            'query': query,
            'max_results': 20
        }
        
        # Extract jurisdiction from query
        jurisdictions = {
            'nsw': ['nsw', 'new south wales'],
            'vic': ['vic', 'victoria'],
            'qld': ['qld', 'queensland'],
            'wa': ['wa', 'western australia'],
            'sa': ['sa', 'south australia'],
            'tas': ['tas', 'tasmania'],
            'act': ['act', 'australian capital territory'],
            'nt': ['nt', 'northern territory']
        }
        
        query_lower = query.lower()
        for juris_code, terms in jurisdictions.items():
            if any(term in query_lower for term in terms):
                params['jurisdiction'] = juris_code
                break
        
        # Extract year if mentioned
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', query)
        if year_match:
            year = int(year_match.group(1))
            params['year_from'] = max(year - 2, 1990)
            params['year_to'] = min(year + 2, datetime.now().year)
        else:
            # Default to recent cases
            params['year_from'] = datetime.now().year - 5
        
        # Override with context if provided
        if context:
            params.update(context)
        
        return params
    
    async def _search_austlii(self, query: str, jurisdiction: str = 'all', 
                             year_from: int = 2020, max_results: int = 20) -> List[Dict]:
        """Search AustLII for cases"""
        cases = []
        
        try:
            # Build search parameters
            search_params = {
                'method': 'boolean',
                'query': query,
                'results': max_results
            }
            
            # Add jurisdiction filter
            if jurisdiction != 'all':
                search_params['meta'] = f'/au/cases/{jurisdiction}/'
            
            # Make search request
            async with self.session.get(
                self.sources['austlii']['search_url'],
                params=search_params
            ) as response:
                if response.status == 200:
                    html = await response.text()
                    cases = self._parse_austlii_search_results(html)
                    
                    # Fetch details for top cases
                    detailed_cases = []
                    for case in cases[:10]:  # Limit to avoid too many requests
                        await asyncio.sleep(self.sources['austlii']['rate_limit'])
                        details = await self._fetch_case_details(case['url'])
                        if details:
                            case.update(details)
                            detailed_cases.append(case)
                    
                    return detailed_cases
                else:
                    logger.warning(f"Search failed with status {response.status}")
                    
        except Exception as e:
            logger.error(f"AustLII search error: {e}")
        
        return cases
    
    def _parse_austlii_search_results(self, html: str) -> List[Dict]:
        """Parse AustLII search results HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        cases = []
        
        # Find all case links
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # AustLII case URLs typically contain /au/cases/
            if '/au/cases/' in href and href.endswith('.html'):
                case_title = link.text.strip()
                
                # Extract citation from title
                citation_match = re.search(r'\[(\d{4})\]\s*([A-Z]+)\s*(\d+)', case_title)
                
                if case_title and citation_match:
                    case = {
                        'title': case_title,
                        'citation': citation_match.group(0),
                        'year': int(citation_match.group(1)),
                        'court': citation_match.group(2),
                        'number': citation_match.group(3),
                        'url': f"{self.sources['austlii']['base_url']}{href}",
                        'source': 'austlii'
                    }
                    cases.append(case)
        
        return cases
    
    async def _fetch_case_details(self, case_url: str) -> Optional[Dict]:
        """Fetch detailed information for a specific case"""
        try:
            async with self.session.get(case_url) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_case_details(html, case_url)
        except Exception as e:
            logger.error(f"Error fetching case details: {e}")
        
        return None
    
    def _parse_case_details(self, html: str, url: str) -> Dict:
        """Parse case details from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        details = {
            'full_text': '\n'.join(lines[:500]),  # First 500 lines
            'url': url,
            'scraped_at': datetime.now().isoformat()
        }
        
        # Try to extract specific fields
        # Extract parties (usually in title or first few lines)
        for line in lines[:10]:
            if ' v ' in line or ' V ' in line:
                details['parties'] = line
                break
        
        # Extract judge
        for line in lines:
            if any(term in line for term in ['Coram:', 'Before:', 'JUDGE:', 'JUSTICE:']):
                details['judge_line'] = line
                break
        
        # Extract catchwords/keywords
        catchwords_start = False
        catchwords = []
        for line in lines:
            if 'CATCHWORDS' in line.upper():
                catchwords_start = True
                continue
            if catchwords_start:
                if line.isupper() or not line:  # End of catchwords section
                    break
                catchwords.append(line)
        
        if catchwords:
            details['catchwords'] = ' '.join(catchwords[:10])  # First 10 lines of catchwords
        
        return details
    
    def _broaden_query(self, query: str) -> str:
        """Broaden a query to find more results"""
        # Remove very specific terms
        broad_query = query
        
        # Remove years
        broad_query = re.sub(r'\b\d{4}\b', '', broad_query)
        
        # Remove specific monetary amounts
        broad_query = re.sub(r'\$[\d,]+', '', broad_query)
        
        # Keep only key legal terms
        key_terms = []
        legal_keywords = [
            'negligence', 'contract', 'breach', 'damage', 'liability',
            'employment', 'dismissal', 'discrimination', 'injury',
            'property', 'criminal', 'appeal', 'settlement'
        ]
        
        for term in query.lower().split():
            if any(keyword in term for keyword in legal_keywords):
                key_terms.append(term)
        
        if key_terms:
            return ' '.join(key_terms)
        
        # If no legal keywords, return first 3 words
        return ' '.join(query.split()[:3])
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate cases"""
        seen = set()
        unique_results = []
        
        for case in results:
            # Create unique identifier
            identifier = case.get('citation', '') or case.get('url', '')
            
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique_results.append(case)
        
        return unique_results
    
    def _rank_results(self, results: List[Dict], query: str, context: Dict = None) -> List[Dict]:
        """Rank results by relevance"""
        query_terms = query.lower().split()
        
        for case in results:
            score = 0
            case_text = f"{case.get('title', '')} {case.get('catchwords', '')} {case.get('full_text', '')}".lower()
            
            # Score based on query term matches
            for term in query_terms:
                score += case_text.count(term)
            
            # Boost recent cases
            if 'year' in case:
                recency_boost = max(0, 5 - (datetime.now().year - case['year']))
                score += recency_boost
            
            # Boost if jurisdiction matches
            if context and context.get('jurisdiction'):
                if context['jurisdiction'] in case.get('url', '').lower():
                    score += 10
            
            case['relevance_score'] = score
        
        # Sort by relevance
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    async def fetch_specific_case(self, citation: str) -> Optional[Dict]:
        """Fetch a specific case by citation"""
        logger.info(f"📋 Fetching specific case: {citation}")
        
        # Search for the exact citation
        results = await self.smart_search(citation, {'max_results': 5})
        
        # Find exact match
        for case in results:
            if case.get('citation') == citation:
                return case
        
        return None
    
    async def monitor_new_cases(self, keywords: List[str], callback=None):
        """Monitor for new cases matching keywords"""
        logger.info(f"👁️ Starting case monitoring for: {keywords}")
        
        last_check = datetime.now()
        
        while True:
            try:
                # Search for cases since last check
                for keyword in keywords:
                    new_cases = await self.smart_search(
                        keyword,
                        {'year_from': last_check.year}
                    )
                    
                    # Filter to only truly new cases
                    for case in new_cases:
                        case_date = self._extract_case_date(case)
                        if case_date and case_date > last_check:
                            logger.info(f"🆕 New case found: {case['citation']}")
                            
                            if callback:
                                await callback(case)
                
                last_check = datetime.now()
                
                # Wait before next check (e.g., daily)
                await asyncio.sleep(86400)  # 24 hours
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    def _extract_case_date(self, case: Dict) -> Optional[datetime]:
        """Extract date from case data"""
        # Try to parse from scraped_at first
        if 'scraped_at' in case:
            try:
                return datetime.fromisoformat(case['scraped_at'])
            except:
                pass
        
        # Try year
        if 'year' in case:
            return datetime(case['year'], 1, 1)
        
        return None

# ===== Integration with Unified System =====

class ScrapingIntegration:
    """Integrate scraping with the unified legal AI system"""
    
    def __init__(self, corpus_manager, min_similarity_threshold=0.7):
        self.corpus = corpus_manager
        self.scraper = None
        self.min_similarity = min_similarity_threshold
        
    async def search_with_scraping(self, query: str, jurisdiction: str = 'all', 
                                  limit: int = 20) -> List[Dict]:
        """
        Search with automatic scraping if not enough results found
        """
        # First, search existing corpus
        existing_results = self.corpus.search_cases(query)
        
        # If we have enough good results, return them
        if len(existing_results) >= limit:
            return existing_results
        
        # Otherwise, scrape for more
        logger.info(f"📊 Only found {len(existing_results)} cases, scraping for more...")
        
        async with IntelligentLegalScraper() as scraper:
            # Scrape additional cases
            scraped_cases = await scraper.smart_search(
                query,
                {'jurisdiction': jurisdiction, 'max_results': limit}
            )
            
            # Convert scraped cases to corpus format
            new_cases = []
            for scraped in scraped_cases:
                case = {
                    'citation': scraped.get('citation', ''),
                    'case_name': scraped.get('title', ''),
                    'year': scraped.get('year', 0),
                    'court': scraped.get('court', ''),
                    'text': scraped.get('full_text', ''),
                    'catchwords': scraped.get('catchwords', ''),
                    'outcome': 'unknown',  # Will need to be analyzed
                    'source': 'scraped',
                    'url': scraped.get('url', '')
                }
                new_cases.append(case)
            
            # Add to corpus for future use
            await self._add_to_corpus(new_cases)
            
            # Combine results
            all_results = existing_results + new_cases
            
            return all_results[:limit]
    
    async def _add_to_corpus(self, cases: List[Dict]):
        """Add new cases to the corpus"""
        # Save to a new file that can be loaded later
        new_cases_file = Path("scraped_cases/new_cases.json")
        new_cases_file.parent.mkdir(exist_ok=True)
        
        existing = []
        if new_cases_file.exists():
            with open(new_cases_file, 'r') as f:
                existing = json.load(f)
        
        existing.extend(cases)
        
        with open(new_cases_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        logger.info(f"💾 Saved {len(cases)} new cases to corpus")

# ===== Example usage =====

async def example_usage():
    """Example of how to use the intelligent scraper"""
    
    async with IntelligentLegalScraper() as scraper:
        # Example 1: Smart search
        print("\n🔍 Example 1: Smart search for negligence cases")
        results = await scraper.smart_search(
            "negligence shopping center injury 2023 NSW"
        )
        
        for case in results[:3]:
            print(f"  - {case['citation']}: {case['title']}")
            if 'catchwords' in case:
                print(f"    Keywords: {case['catchwords'][:100]}...")
        
        # Example 2: Fetch specific case
        print("\n📋 Example 2: Fetch specific case")
        case = await scraper.fetch_specific_case("[2023] NSWSC 100")
        if case:
            print(f"  Found: {case['title']}")
        
        # Example 3: Monitor for new cases (commented out as it runs forever)
        # print("\n👁️ Example 3: Monitor for new cases")
        # await scraper.monitor_new_cases(
        #     keywords=['artificial intelligence', 'AI liability'],
        #     callback=lambda case: print(f"New AI case: {case['citation']}")
        # )

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
