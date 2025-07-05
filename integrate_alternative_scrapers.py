#!/usr/bin/env python3
"""
Integration code to add alternative scrapers to your unified system
"""

print('''
# Add this import to your unified_with_scraping.py:
from alternative_scrapers import AlternativeScrapers

# Update the IntelligentLegalScraper class to include alternative sources:

# In the smart_search method, add after AustLII attempt:

        # If AustLII fails or returns too few results, try alternative sources
        if len(results) < 10:
            logger.info("📊 Trying alternative sources...")
            
            async with AlternativeScrapers() as alt_scrapers:
                alt_results = await alt_scrapers.scrape_all_sources(
                    query=search_params['query'],
                    max_per_source=5
                )
                
                # Convert to standard format
                for alt_case in alt_results:
                    standardized = {
                        'title': alt_case.get('title', ''),
                        'citation': alt_case.get('citation', ''),
                        'year': alt_case.get('year', 0),
                        'court': alt_case.get('court', ''),
                        'url': alt_case.get('url', ''),
                        'source': alt_case.get('source', 'alternative'),
                        'catchwords': alt_case.get('catchwords', ''),
                        'summary': alt_case.get('summary', '')
                    }
                    
                    # Only add if not duplicate
                    if not any(r['citation'] == standardized['citation'] for r in results):
                        results.append(standardized)

# Add new endpoint for testing alternative scrapers:

@app.post("/api/v1/scrape/alternatives")
async def scrape_alternative_sources(query: str, source: str = "all"):
    """Scrape from alternative sources (Federal Court, High Court, etc.)"""
    try:
        async with AlternativeScrapers() as scrapers:
            if source == "federal":
                cases = await scrapers.scrape_federal_court(query)
            elif source == "high":
                cases = await scrapers.scrape_high_court(query)
            elif source == "nsw":
                cases = await scrapers.scrape_nsw_caselaw(query)
            elif source == "commonwealth":
                cases = await scrapers.scrape_comcourts(query)
            else:
                cases = await scrapers.scrape_all_sources(query)
            
            return {
                "status": "success",
                "source": source,
                "query": query,
                "cases_found": len(cases),
                "cases": cases
            }
            
    except Exception as e:
        logger.error(f"Alternative scraping error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
''')
