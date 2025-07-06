#!/usr/bin/env python3
"""
Enhanced Australian Legal AI - Working Version with AI Integration
"""
import json
import re
import requests
import asyncio
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🧠 Enhanced Australian Legal AI",
    description="Real legal data with AI-powered analysis",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for legal corpus
legal_corpus = []
keyword_index = defaultdict(set)
metadata_index = {}

class SmartSearchRequest(BaseModel):
    query: str
    num_results: int = 10
    search_type: str = "hybrid"
    jurisdiction: str = None
    document_type: str = None
    use_ai_analysis: bool = True

class AIAnalysisRequest(BaseModel):
    text: str
    analysis_types: List[str] = ["summary", "entities", "classification", "concepts"]

class LegalResearchRequest(BaseModel):
    legal_question: str
    context: str = ""

def load_legal_corpus():
    """Load the real Australian legal corpus"""
    global legal_corpus, keyword_index, metadata_index
    
    try:
        logger.info("Loading Australian legal corpus...")
        with open('corpus_export/australian_legal_corpus.jsonl', 'r') as f:
            for i, line in enumerate(f):
                if i >= 1000:  # Load 1000 docs for performance
                    break
                    
                doc = json.loads(line.strip())
                legal_corpus.append(doc)
                
                # Build keyword index
                text = doc['text'].lower()
                words = re.findall(r'\b\w+\b', text)
                for word in words:
                    if len(word) > 3:
                        keyword_index[word].add(i)
                
                metadata_index[i] = doc.get('metadata', {})
        
        logger.info(f"Loaded {len(legal_corpus)} legal documents")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load legal corpus: {e}")
        return False

def search_documents_enhanced(query: str, num_results: int = 10, filters: Dict = None) -> List[Dict]:
    """Enhanced search with better scoring"""
    
    if not legal_corpus:
        return []
    
    query_words = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 2]
    doc_scores = Counter()
    
    # Enhanced scoring with phrase matching and legal term weighting
    legal_weight_terms = {
        'contract', 'tort', 'negligence', 'duty', 'care', 'breach', 'damages',
        'law', 'act', 'section', 'court', 'judge', 'case', 'appeal', 'plaintiff',
        'defendant', 'jurisdiction', 'statute', 'regulation', 'precedent',
        'liability', 'remedy', 'constitutional', 'criminal', 'civil'
    }
    
    for word in query_words:
        if word in keyword_index:
            weight = 3 if word in legal_weight_terms else 1
            for doc_id in keyword_index[word]:
                doc_text = legal_corpus[doc_id]['text'].lower()
                word_count = doc_text.count(word)
                doc_scores[doc_id] += word_count * weight
    
    # Bonus for phrase matches
    query_phrase = ' '.join(query_words)
    for doc_id in range(len(legal_corpus)):
        doc_text = legal_corpus[doc_id]['text'].lower()
        if query_phrase in doc_text:
            doc_scores[doc_id] += 10
    
    # Apply filters
    if filters:
        filtered_scores = {}
        for doc_id, score in doc_scores.items():
            metadata = metadata_index.get(doc_id, {})
            include = True
            
            if filters.get('jurisdiction') and metadata.get('jurisdiction') != filters['jurisdiction']:
                include = False
            if filters.get('document_type') and metadata.get('type') != filters['document_type']:
                include = False
                
            if include:
                filtered_scores[doc_id] = score
        doc_scores = Counter(filtered_scores)
    
    # Get results with enhanced metadata
    results = []
    for doc_id, score in doc_scores.most_common(num_results):
        doc = legal_corpus[doc_id]
        text = doc['text']
        
        # Create enhanced snippet with context
        snippet = create_smart_snippet(text, query_words)
        
        # Calculate relevance score
        max_possible_score = len(query_words) * 15
        relevance = min(score / max(max_possible_score, 1), 1.0)
        
        results.append({
            "document_id": doc_id,
            "text": text,
            "snippet": snippet,
            "metadata": doc.get('metadata', {}),
            "citation": doc.get('metadata', {}).get('citation', 'Australian Legal Document'),
            "relevance_score": round(relevance, 3),
            "match_count": score,
            "search_method": "enhanced_keyword"
        })
    
    return results

def create_smart_snippet(text: str, query_words: List[str], max_length: int = 400) -> str:
    """Create intelligent snippet showing relevant context"""
    
    text_lower = text.lower()
    
    # Find best sentence with query terms
    sentences = re.split(r'[.!?]+', text)
    best_sentence_idx = 0
    best_score = 0
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        score = sum(1 for word in query_words if word in sentence_lower)
        if score > best_score:
            best_score = score
            best_sentence_idx = i
    
    # Get context around best sentence
    start_idx = max(0, best_sentence_idx - 1)
    end_idx = min(len(sentences), best_sentence_idx + 2)
    
    snippet = '. '.join(sentences[start_idx:end_idx]).strip()
    
    if len(snippet) > max_length:
        snippet = snippet[:max_length] + "..."
    
    return snippet

def analyze_legal_concepts_enhanced(text: str) -> Dict[str, Any]:
    """Enhanced legal concept analysis"""
    
    # Enhanced legal concept patterns
    legal_patterns = {
        "contract_law": [
            r'\b(?:offer|acceptance|consideration|intention|breach|damages|termination)\b',
            r'\b(?:warranty|condition|term|clause|contract|agreement|privity)\b',
            r'\b(?:specific performance|liquidated damages|penalty|frustration)\b'
        ],
        "tort_law": [
            r'\b(?:negligence|duty of care|breach|causation|remoteness|foreseeability)\b',
            r'\b(?:damages|injury|harm|loss|novus actus interveniens)\b',
            r'\b(?:reasonable person|standard of care|contributory negligence|volenti)\b'
        ],
        "constitutional_law": [
            r'\b(?:constitution|constitutional|section 51|section 92|section 109)\b',
            r'\b(?:separation of powers|judicial review|executive|legislative)\b',
            r'\b(?:implied freedom|characterisation|external affairs|trade and commerce)\b'
        ],
        "criminal_law": [
            r'\b(?:mens rea|actus reus|intent|recklessness|negligence|strict liability)\b',
            r'\b(?:charge|conviction|sentence|penalty|defence|mitigation)\b',
            r'\b(?:beyond reasonable doubt|burden of proof|presumption)\b'
        ],
        "evidence_law": [
            r'\b(?:evidence|witness|testimony|hearsay|relevance|probative)\b',
            r'\b(?:admissible|inadmissible|privilege|similar fact|character)\b'
        ],
        "corporate_law": [
            r'\b(?:director|shareholder|fiduciary|business judgment|oppression)\b',
            r'\b(?:corporation|company|board|dividend|piercing corporate veil)\b'
        ]
    }
    
    # Citation extraction patterns
    citation_patterns = [
        r'\[(\d{4})\]\s+([A-Z]+)\s+(\d+)',  # [2023] HCA 15
        r'\((\d{4})\)\s+(\d+)\s+([A-Z]+)\s+(\d+)',  # (2023) 97 ALJR 123
        r'([A-Z][A-Za-z\s&]+)\s+v\s+([A-Z][A-Za-z\s&]+)',  # Case v Case
        r'\b([A-Z][a-z]+\s+Act\s+\d{4})\b',  # Privacy Act 1988
        r'\bs\s*(\d+[A-Z]*)\b'  # s 109, s 51(vi)
    ]
    
    analysis = {
        "legal_areas": {},
        "citations": [],
        "statutory_references": [],
        "key_concepts": [],
        "complexity_score": 0,
        "word_count": len(text.split())
    }
    
    text_lower = text.lower()
    
    # Analyze legal areas
    total_legal_terms = 0
    for area, patterns in legal_patterns.items():
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text_lower, re.IGNORECASE)
            matches.extend(found)
        
        if matches:
            unique_matches = list(set(matches))
            analysis["legal_areas"][area] = {
                "terms_found": unique_matches,
                "frequency": len(matches),
                "confidence": min(len(unique_matches) / 5, 1.0),
                "density": len(matches) / len(text.split())
            }
            total_legal_terms += len(matches)
    
    # Extract citations and references
    for pattern in citation_patterns:
        citations = re.findall(pattern, text, re.IGNORECASE)
        if citations:
            analysis["citations"].extend([str(c) for c in citations])
    
    # Calculate complexity
    analysis["complexity_score"] = min(total_legal_terms / max(len(text.split()), 1) * 1000, 100)
    
    # Key concepts by frequency
    all_matches = []
    for area_data in analysis["legal_areas"].values():
        all_matches.extend(area_data["terms_found"])
    
    concept_freq = Counter(all_matches)
    analysis["key_concepts"] = [
        {"concept": concept, "frequency": freq}
        for concept, freq in concept_freq.most_common(8)
    ]
    
    return analysis

async def get_ai_summary_simple(text: str) -> str:
    """Simple AI summary using local analysis"""
    
    # Extract key sentences
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) <= 3:
        return text[:200] + "..." if len(text) > 200 else text
    
    # Score sentences by legal importance
    legal_keywords = {
        'held', 'court', 'judge', 'ruling', 'decision', 'case', 'law', 'act',
        'section', 'plaintiff', 'defendant', 'appeal', 'jurisdiction', 'liability',
        'damages', 'breach', 'duty', 'negligence', 'contract', 'tort'
    }
    
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        score = sum(1 for word in legal_keywords if word in sentence_lower)
        score += 2 if i < 3 else 0  # Boost early sentences
        sentence_scores.append((score, sentence.strip()))
    
    # Get top 2-3 sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:3]
    summary = '. '.join([s[1] for s in top_sentences if s[1]])
    
    return summary[:300] + "..." if len(summary) > 300 else summary

# Startup
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Enhanced Australian Legal AI...")
    corpus_loaded = load_legal_corpus()
    
    if corpus_loaded:
        logger.info("✅ Enhanced Legal AI ready!")
    else:
        logger.error("❌ Failed to initialize Legal AI")

# Routes
@app.get("/")
def root():
    return FileResponse("static/smart_index.html")

@app.get("/api")
def api_info():
    return {
        "name": "🧠 Enhanced Australian Legal AI",
        "status": "operational",
        "version": "3.1.0",
        "corpus_size": len(legal_corpus),
        "ai_features": [
            "Enhanced legal search",
            "Legal concept analysis",
            "Smart snippet generation",
            "Citation extraction",
            "Legal area classification"
        ],
        "models_loaded": {
            "enhanced_search": True,
            "concept_analysis": True,
            "citation_extraction": True,
            "legal_classification": True
        }
    }

@app.post("/api/v1/smart-search")
async def smart_search(request: SmartSearchRequest):
    """Enhanced smart search"""
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Prepare filters
    filters = {}
    if request.jurisdiction:
        filters['jurisdiction'] = request.jurisdiction.lower()
    if request.document_type:
        filters['document_type'] = request.document_type.lower()
    
    # Perform enhanced search
    results = search_documents_enhanced(request.query, request.num_results, filters)
    
    # Add AI analysis if requested
    query_analysis = {}
    if request.use_ai_analysis:
        query_analysis = analyze_legal_concepts_enhanced(request.query)
        
        # Enhance top results with AI analysis
        for result in results[:3]:
            try:
                doc_analysis = analyze_legal_concepts_enhanced(result['text'][:1000])
                ai_summary = await get_ai_summary_simple(result['text'][:800])
                
                result['ai_analysis'] = {
                    "summary": ai_summary,
                    "legal_areas": doc_analysis['legal_areas'],
                    "key_concepts": doc_analysis['key_concepts'][:5],
                    "complexity": round(doc_analysis['complexity_score'], 1),
                    "citations": doc_analysis['citations'][:3]
                }
            except Exception as e:
                logger.warning(f"AI analysis failed for doc {result['document_id']}: {e}")
    
    return {
        "status": "success",
        "query": request.query,
        "search_type": "enhanced",
        "total_results": len(results),
        "filters_applied": filters,
        "query_analysis": query_analysis,
        "results": results
    }

@app.post("/api/v1/ai-analysis")
async def ai_analysis(request: AIAnalysisRequest):
    """AI analysis of legal text"""
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    analysis_results = {}
    
    # Always include enhanced concept analysis
    analysis_results["concepts"] = analyze_legal_concepts_enhanced(request.text)
    
    # Add summary if requested
    if "summary" in request.analysis_types:
        analysis_results["summary"] = await get_ai_summary_simple(request.text)
    
    # Add entities (simplified)
    if "entities" in request.analysis_types:
        # Extract person/organization names using simple patterns
        person_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+J\b'  # Judge names
        org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Pty\s+Ltd|Ltd|Corporation|Inc))\b'
        
        persons = re.findall(person_pattern, request.text)
        orgs = re.findall(org_pattern, request.text)
        
        analysis_results["entities"] = {
            "persons": list(set(persons)),
            "organizations": list(set(orgs))
        }
    
    return {
        "status": "success",
        "text_length": len(request.text),
        "analysis": analysis_results
    }

@app.post("/api/v1/legal-research")
async def legal_research(request: LegalResearchRequest):
    """Enhanced legal research"""
    
    # Search for relevant documents
    search_results = search_documents_enhanced(request.legal_question, 15)
    
    # Analyze the question
    question_analysis = analyze_legal_concepts_enhanced(request.legal_question)
    
    # Group results by legal area
    results_by_area = defaultdict(list)
    for result in search_results:
        doc_analysis = analyze_legal_concepts_enhanced(result['text'][:500])
        
        if doc_analysis['legal_areas']:
            primary_area = max(
                doc_analysis['legal_areas'].items(),
                key=lambda x: x[1]['confidence']
            )[0]
        else:
            primary_area = 'general'
            
        results_by_area[primary_area].append(result)
    
    # Create research summary
    research_summary = {
        "question": request.legal_question,
        "legal_areas_identified": list(question_analysis['legal_areas'].keys()),
        "total_documents_found": len(search_results),
        "results_by_area": dict(results_by_area),
        "key_cases": [r for r in search_results[:5] if 'decision' in r.get('metadata', {}).get('type', '')],
        "relevant_legislation": [r for r in search_results[:5] if 'legislation' in r.get('metadata', {}).get('type', '')],
        "research_confidence": min(len(search_results) / 10, 1.0)
    }
    
    return {
        "status": "success",
        "research_summary": research_summary,
        "question_analysis": question_analysis
    }

@app.get("/api/v1/stats")
def get_enhanced_stats():
    """Enhanced statistics"""
    
    jurisdictions = Counter()
    doc_types = Counter()
    
    for doc_id, metadata in metadata_index.items():
        jurisdictions[metadata.get('jurisdiction', 'unknown')] += 1
        doc_types[metadata.get('type', 'unknown')] += 1
    
    return {
        "corpus_info": {
            "total_documents": len(legal_corpus),
            "total_keywords": len(keyword_index),
            "ai_enhanced": True
        },
        "ai_models": {
            "enhanced_search": True,
            "concept_analysis": True,
            "citation_extraction": True,
            "smart_summarization": True
        },
        "corpus_breakdown": {
            "jurisdictions": dict(jurisdictions.most_common()),
            "document_types": dict(doc_types.most_common())
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)