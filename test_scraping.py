#!/usr/bin/env python3
"""Test the intelligent scraping features"""

import asyncio
from intelligent_legal_scraper import IntelligentLegalScraper

async def test_scraping():
    print("🧪 Testing Intelligent Legal Scraper")
    print("=" * 60)
    
    async with IntelligentLegalScraper() as scraper:
        # Test 1: Search for recent cases
        print("\n1️⃣ Searching for recent negligence cases...")
        results = await scraper.smart_search(
            "negligence personal injury NSW 2023",
            {'max_results': 5}
        )
        
        print(f"Found {len(results)} cases:")
        for case in results:
            print(f"  - {case.get('citation', 'Unknown')}: {case.get('title', '')[:60]}...")
            if case.get('catchwords'):
                print(f"    Keywords: {case['catchwords'][:80]}...")
        
        # Test 2: Broaden search
        print("\n2️⃣ Testing query broadening...")
        original = "specific negligence case $2,500,000 damages 2023 NSW Supreme Court"
        broadened = scraper._broaden_query(original)
        print(f"  Original: {original}")
        print(f"  Broadened: {broadened}")
        
        # Test 3: Search analysis
        print("\n3️⃣ Testing search parameter analysis...")
        params = scraper._analyze_search_needs(
            "contract breach Melbourne 2022 construction"
        )
        print(f"  Extracted parameters: {params}")

if __name__ == "__main__":
    asyncio.run(test_scraping())
