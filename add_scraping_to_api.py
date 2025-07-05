#!/usr/bin/env python3
"""
Add these endpoints to your unified_legal_ai_system_fixed.py
"""

print('''
Add these to your unified_legal_ai_system_fixed.py:

# Import at the top
from intelligent_legal_scraper import IntelligentLegalScraper, ScrapingIntegration

# Add to UnifiedLegalAI.__init__:
        self.scraping_integration = ScrapingIntegration(self.corpus)
        self.auto_scrape_enabled = True

# Add these new endpoints:

@app.post("/api/v1/search/smart")
async def smart_search_with_scraping(request: SearchRequest):
    """Search with automatic scraping if needed"""
    try:
        # First try normal search
        results = await unified_ai.search_cases(
            query=request.query,
            jurisdiction=request.jurisdiction,
            limit=request.limit
        )
        
        # If not enough results and auto-scrape is enabled
        if len(results) < 5 and unified_ai.auto_scrape_enabled:
            logger.info(f"Only {len(results)} results found, triggering smart scraping...")
            
            # Use scraping integration
            all_results = await unified_ai.scraping_integration.search_with_scraping(
                query=request.query,
                jurisdiction=request.jurisdiction,
                limit=request.limit
            )
            
            # Format results
            formatted_results = []
            for r in all_results:
                if isinstance(r, dict) and 'citation' in r:
                    # Scraped result
                    formatted_results.append({
                        "citation": r['citation'],
                        "case_name": r.get('case_name', r.get('title', '')),
                        "year": r.get('year', 0),
                        "court": r.get('court', ''),
                        "source": r.get('source', 'corpus'),
                        "snippet": r.get('text', '')[:200] + "...",
                        "url": r.get('url', '')
                    })
            
            return {
                "query": request.query,
                "search_type": "smart_scraping",
                "results_count": len(formatted_results),
                "scraped_new": len([r for r in formatted_results if r['source'] == 'scraped']),
                "results": formatted_results
            }
        else:
            # Normal results
            return await search_cases(request)
            
    except Exception as e:
        logger.error(f"Smart search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/scrape/case")
async def scrape_specific_case(citation: str):
    """Scrape a specific case by citation"""
    try:
        async with IntelligentLegalScraper() as scraper:
            case = await scraper.fetch_specific_case(citation)
            
            if case:
                # Add to corpus
                await unified_ai.scraping_integration._add_to_corpus([case])
                
                return {
                    "status": "success",
                    "case": case
                }
            else:
                raise HTTPException(status_code=404, detail=f"Case {citation} not found")
                
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/scrape/topic")
async def scrape_topic(topic: str, max_cases: int = 20):
    """Scrape cases about a specific topic"""
    try:
        async with IntelligentLegalScraper() as scraper:
            cases = await scraper.smart_search(topic, {'max_results': max_cases})
            
            # Add to corpus
            if cases:
                await unified_ai.scraping_integration._add_to_corpus(cases)
            
            return {
                "status": "success",
                "topic": topic,
                "cases_found": len(cases),
                "cases": cases
            }
            
    except Exception as e:
        logger.error(f"Topic scraping error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/scrape/status")
async def get_scraping_status():
    """Get scraping status and statistics"""
    scraped_dir = Path("scraped_cases")
    scraped_count = 0
    
    if scraped_dir.exists():
        for file in scraped_dir.glob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                scraped_count += len(data) if isinstance(data, list) else 1
    
    return {
        "auto_scrape_enabled": unified_ai.auto_scrape_enabled,
        "scraped_cases_count": scraped_count,
        "corpus_size": len(unified_ai.corpus.cases),
        "total_available": len(unified_ai.corpus.cases) + scraped_count
    }

@app.post("/api/v1/scrape/toggle")
async def toggle_auto_scraping(enabled: bool):
    """Enable or disable automatic scraping"""
    unified_ai.auto_scrape_enabled = enabled
    return {
        "status": "success",
        "auto_scrape_enabled": enabled
    }

# Background task for monitoring
@app.on_event("startup")
async def startup_event():
    """Start background monitoring if configured"""
    # Example: Monitor for new AI-related cases
    # asyncio.create_task(monitor_topics(['artificial intelligence law', 'AI liability']))
    pass
''')
