#!/usr/bin/env python3
"""
Smart Australian Legal AI with Hugging Face Integration
Advanced legal analysis, semantic search, and AI reasoning
"""
import json
import re
import requests
import asyncio
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import logging
from datetime import datetime
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🧠 Smart Australian Legal AI",
    description="AI-powered legal research with semantic search and advanced analysis",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global AI models and data
legal_corpus = []
keyword_index = defaultdict(set)
metadata_index = {}
embeddings_model = None
legal_embeddings = None
summarizer = None
legal_classifier = None
ner_pipeline = None

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co"
HF_HEADERS = {"Authorization": "Bearer hf_demo"}  # Use demo token, user should add their own

class SmartSearchRequest(BaseModel):
    query: str
    num_results: int = 10
    search_type: str = "hybrid"  # keyword, semantic, hybrid
    jurisdiction: str = None
    document_type: str = None
    use_ai_analysis: bool = True

class AIAnalysisRequest(BaseModel):
    text: str
    analysis_types: List[str] = ["summary", "entities", "classification", "concepts"]
    include_precedents: bool = True
    
class LegalResearchRequest(BaseModel):
    legal_question: str
    context: str = ""
    research_depth: str = "standard"  # basic, standard, comprehensive
    
class CaseAnalysisRequest(BaseModel):
    case_facts: str
    legal_issues: List[str]
    jurisdiction: str = "commonwealth"
    
def init_ai_models():
    """Initialize AI models for legal analysis"""
    global embeddings_model, summarizer, legal_classifier, ner_pipeline
    
    try:
        logger.info("Loading AI models...")
        
        # Sentence transformer for semantic search
        logger.info("Loading sentence transformer...")
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Legal text summarization
        logger.info("Loading summarization pipeline...")
        summarizer = pipeline("summarization", 
                            model="facebook/bart-base",
                            device=-1)  # CPU
        
        # Legal text classification
        logger.info("Loading classification pipeline...")
        legal_classifier = pipeline("text-classification",
                                   model="distilbert-base-uncased",
                                   device=-1)
        
        # Named Entity Recognition for legal entities
        logger.info("Loading NER pipeline...")
        ner_pipeline = pipeline("ner", 
                               model="dbmdz/bert-large-cased-finetuned-conll03-english",
                               aggregation_strategy="simple",
                               device=-1)
        
        logger.info("✅ All AI models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load AI models: {e}")
        logger.warning("🔄 Falling back to API-based models...")
        return False

def load_legal_corpus_with_embeddings():
    """Load legal corpus and generate embeddings"""
    global legal_corpus, keyword_index, metadata_index, legal_embeddings
    
    try:
        logger.info("Loading legal corpus with AI embeddings...")
        
        # Load corpus
        with open('corpus_export/australian_legal_corpus.jsonl', 'r') as f:
            texts_for_embedding = []
            
            for i, line in enumerate(f):
                if i >= 500:  # Limit for demo
                    break
                    
                doc = json.loads(line.strip())
                legal_corpus.append(doc)
                
                # Prepare text for embedding
                text_snippet = doc['text'][:512]  # Limit length for embeddings
                texts_for_embedding.append(text_snippet)
                
                # Build keyword index
                text = doc['text'].lower()
                words = re.findall(r'\b\w+\b', text)
                for word in words:
                    if len(word) > 3:
                        keyword_index[word].add(i)
                
                metadata_index[i] = doc.get('metadata', {})
        
        # Generate embeddings if model is available
        if embeddings_model:
            logger.info("Generating semantic embeddings...")
            legal_embeddings = embeddings_model.encode(texts_for_embedding)
            logger.info(f"✅ Generated embeddings for {len(legal_embeddings)} documents")
        
        logger.info(f"📚 Loaded {len(legal_corpus)} legal documents")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        return False

def semantic_search(query: str, num_results: int = 10) -> List[Dict]:
    """Semantic search using embeddings"""
    if not embeddings_model or legal_embeddings is None:
        return []
    
    try:
        # Get query embedding
        query_embedding = embeddings_model.encode([query])
        
        # Calculate similarities
        similarities = np.dot(legal_embeddings, query_embedding.T).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:num_results]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                doc = legal_corpus[idx]
                results.append({
                    "document_id": idx,
                    "text": doc['text'],
                    "snippet": doc['text'][:500] + "...",
                    "metadata": doc.get('metadata', {}),
                    "citation": doc.get('metadata', {}).get('citation', 'Australian Legal Document'),
                    "semantic_score": float(similarities[idx]),
                    "relevance_score": min(float(similarities[idx]) * 2, 1.0)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []

def keyword_search(query: str, num_results: int = 10, filters: Dict = None) -> List[Dict]:
    """Enhanced keyword search from original system"""
    if not legal_corpus:
        return []
    
    query_words = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 2]
    doc_scores = Counter()
    
    # Score documents
    for word in query_words:
        if word in keyword_index:
            for doc_id in keyword_index[word]:
                doc_text = legal_corpus[doc_id]['text'].lower()
                word_count = doc_text.count(word)
                doc_scores[doc_id] += word_count
    
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
    
    # Get results
    results = []
    for doc_id, score in doc_scores.most_common(num_results):
        doc = legal_corpus[doc_id]
        text = doc['text']
        snippet = text[:500] + "..." if len(text) > 500 else text
        
        max_possible_score = len(query_words) * 10
        relevance = min(score / max(max_possible_score, 1), 1.0)
        
        results.append({
            "document_id": doc_id,
            "text": text,
            "snippet": snippet,
            "metadata": doc.get('metadata', {}),
            "citation": doc.get('metadata', {}).get('citation', 'Australian Legal Document'),
            "relevance_score": round(relevance, 3),
            "match_count": score
        })
    
    return results

def hybrid_search(query: str, num_results: int = 10, filters: Dict = None) -> List[Dict]:
    """Combine semantic and keyword search"""
    
    # Get semantic results
    semantic_results = semantic_search(query, num_results * 2)
    
    # Get keyword results  
    keyword_results = keyword_search(query, num_results * 2, filters)
    
    # Combine and deduplicate
    combined_docs = {}
    
    # Add semantic results
    for result in semantic_results:
        doc_id = result['document_id']
        result['search_method'] = 'semantic'
        combined_docs[doc_id] = result
    
    # Add keyword results, merge scores if duplicate
    for result in keyword_results:
        doc_id = result['document_id']
        if doc_id in combined_docs:
            # Combine scores
            existing = combined_docs[doc_id]
            existing['relevance_score'] = max(
                existing.get('relevance_score', 0),
                result['relevance_score']
            )
            existing['search_method'] = 'hybrid'
            if 'match_count' in result:
                existing['match_count'] = result['match_count']
        else:
            result['search_method'] = 'keyword'
            combined_docs[doc_id] = result
    
    # Sort by relevance and return top results
    sorted_results = sorted(
        combined_docs.values(),
        key=lambda x: x.get('relevance_score', 0),
        reverse=True
    )
    
    return sorted_results[:num_results]

async def analyze_with_huggingface(text: str, task: str) -> Dict[str, Any]:
    """Use Hugging Face API for advanced analysis"""
    
    task_configs = {
        "summarization": {
            "url": f"{HF_API_URL}/models/facebook/bart-large-cnn",
            "payload": {"inputs": text[:1024]}  # Limit input length
        },
        "classification": {
            "url": f"{HF_API_URL}/models/cardiffnlp/twitter-roberta-base-sentiment-latest", 
            "payload": {"inputs": text[:512]}
        },
        "question_answering": {
            "url": f"{HF_API_URL}/models/deepset/roberta-base-squad2",
            "payload": {
                "inputs": {
                    "question": "What is the main legal issue?",
                    "context": text[:1024]
                }
            }
        }
    }
    
    if task not in task_configs:
        return {"error": f"Task {task} not supported"}
    
    try:
        config = task_configs[task]
        response = requests.post(
            config["url"],
            headers=HF_HEADERS,
            json=config["payload"],
            timeout=10
        )
        
        if response.status_code == 200:
            return {"result": response.json(), "source": "huggingface_api"}
        else:
            return {"error": f"API error: {response.status_code}"}
            
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def analyze_legal_concepts(text: str) -> Dict[str, Any]:
    """Extract legal concepts and entities"""
    
    # Legal concept patterns (enhanced)
    legal_patterns = {
        "contract_law": [
            r'\b(?:offer|acceptance|consideration|intention|breach|damages|termination|rescission)\b',
            r'\b(?:warranty|condition|term|clause|contract|agreement|party|parties)\b',
            r'\b(?:specific performance|liquidated damages|penalty clause|force majeure)\b'
        ],
        "tort_law": [
            r'\b(?:negligence|duty of care|breach|causation|proximity|foreseeability)\b',
            r'\b(?:damages|injury|harm|loss|but for test|novus actus interveniens)\b',
            r'\b(?:reasonable person|standard of care|contributory negligence)\b'
        ],
        "corporate_law": [
            r'\b(?:director|shareholder|fiduciary duty|business judgment rule)\b',
            r'\b(?:corporation|company|board|meeting|resolution|dividend)\b',
            r'\b(?:piercing corporate veil|oppression|derivative action)\b'
        ],
        "criminal_law": [
            r'\b(?:mens rea|actus reus|intent|recklessness|negligence|strict liability)\b',
            r'\b(?:charge|conviction|sentence|penalty|defence|mitigation)\b',
            r'\b(?:beyond reasonable doubt|burden of proof|presumption of innocence)\b'
        ],
        "constitutional_law": [
            r'\b(?:constitution|constitutional|section 51|section 92|section 109)\b',
            r'\b(?:separation of powers|judicial review|executive power|legislative power)\b',
            r'\b(?:implied freedom|characterisation|external affairs power)\b'
        ],
        "evidence_law": [
            r'\b(?:evidence|witness|testimony|hearsay|relevance|probative value)\b',
            r'\b(?:objection|admissible|inadmissible|privilege|public interest immunity)\b',
            r'\b(?:similar fact evidence|character evidence|opinion evidence)\b'
        ]
    }
    
    # Citation patterns
    citation_patterns = [
        r'\[(\d{4})\]\s+([A-Z]+)\s+(\d+)',  # [2023] HCA 15
        r'\((\d{4})\)\s+(\d+)\s+([A-Z]+)\s+(\d+)',  # (2023) 97 ALJR 123
        r'([A-Z][A-Za-z\s&]+)\s+v\s+([A-Z][A-Za-z\s&]+)\s+\[(\d{4})\]'  # Case v Case [2023]
    ]
    
    analysis = {
        "legal_areas": {},
        "citations": [],
        "legal_entities": [],
        "complexity_score": 0,
        "key_concepts": []
    }
    
    text_lower = text.lower()
    
    # Find legal concepts by area
    for area, patterns in legal_patterns.items():
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text_lower, re.IGNORECASE)
            matches.extend(found)
        
        if matches:
            analysis["legal_areas"][area] = {
                "terms_found": list(set(matches)),
                "frequency": len(matches),
                "confidence": min(len(set(matches)) / 10, 1.0)
            }
    
    # Extract citations
    for pattern in citation_patterns:
        citations = re.findall(pattern, text)
        analysis["citations"].extend(citations)
    
    # Use NER if available
    if ner_pipeline:
        try:
            entities = ner_pipeline(text[:512])  # Limit length
            legal_entities = [
                ent for ent in entities 
                if ent['entity_group'] in ['PER', 'ORG', 'LOC'] and ent['score'] > 0.9
            ]
            analysis["legal_entities"] = legal_entities
        except:
            pass
    
    # Calculate complexity
    total_concepts = sum(
        area["frequency"] for area in analysis["legal_areas"].values()
    )
    word_count = len(text.split())
    analysis["complexity_score"] = min(total_concepts / max(word_count, 1) * 1000, 100)
    
    # Extract key concepts
    all_matches = []
    for area_data in analysis["legal_areas"].values():
        all_matches.extend(area_data["terms_found"])
    
    concept_freq = Counter(all_matches)
    analysis["key_concepts"] = [
        {"concept": concept, "frequency": freq}
        for concept, freq in concept_freq.most_common(10)
    ]
    
    return analysis

# Startup
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Smart Australian Legal AI...")
    
    # Initialize AI models
    models_loaded = init_ai_models()
    
    # Load corpus
    corpus_loaded = load_legal_corpus_with_embeddings()
    
    if models_loaded and corpus_loaded:
        logger.info("✅ Smart Legal AI ready with full AI capabilities!")
    elif corpus_loaded:
        logger.info("⚠️ Legal AI ready with basic search (AI models failed to load)")
    else:
        logger.error("❌ Failed to initialize Legal AI")

# Routes
@app.get("/")
def root():
    return FileResponse("static/smart_index.html")

@app.get("/api")
def api_info():
    return {
        "name": "🧠 Smart Australian Legal AI",
        "status": "operational",
        "version": "3.0.0",
        "corpus_size": len(legal_corpus),
        "ai_features": [
            "Semantic search with embeddings",
            "Legal concept extraction", 
            "AI-powered summarization",
            "Named entity recognition",
            "Case law analysis",
            "Precedent research"
        ],
        "models_loaded": {
            "embeddings": embeddings_model is not None,
            "summarizer": summarizer is not None,
            "classifier": legal_classifier is not None,
            "ner": ner_pipeline is not None
        }
    }

@app.post("/api/v1/smart-search")
async def smart_search(request: SmartSearchRequest):
    """Intelligent search with AI analysis"""
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Prepare filters
    filters = {}
    if request.jurisdiction:
        filters['jurisdiction'] = request.jurisdiction.lower()
    if request.document_type:
        filters['document_type'] = request.document_type.lower()
    
    # Perform search based on type
    if request.search_type == "semantic" and embeddings_model:
        results = semantic_search(request.query, request.num_results)
    elif request.search_type == "keyword":
        results = keyword_search(request.query, request.num_results, filters)
    else:  # hybrid
        results = hybrid_search(request.query, request.num_results, filters)
    
    # Add AI analysis if requested
    if request.use_ai_analysis and results:
        # Analyze the query
        query_analysis = analyze_legal_concepts(request.query)
        
        # Enhance results with AI insights
        for result in results[:3]:  # Analyze top 3 results
            try:
                doc_analysis = analyze_legal_concepts(result['text'][:1000])
                result['ai_analysis'] = {
                    "legal_areas": doc_analysis['legal_areas'],
                    "key_concepts": doc_analysis['key_concepts'][:5],
                    "complexity": round(doc_analysis['complexity_score'], 1)
                }
            except:
                pass
    else:
        query_analysis = {}
    
    return {
        "status": "success",
        "query": request.query,
        "search_type": request.search_type,
        "total_results": len(results),
        "filters_applied": filters,
        "query_analysis": query_analysis,
        "results": results
    }

@app.post("/api/v1/ai-analysis")
async def ai_analysis(request: AIAnalysisRequest):
    """Comprehensive AI analysis of legal text"""
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    analysis_results = {}
    
    # Legal concept analysis (always included)
    analysis_results["concepts"] = analyze_legal_concepts(request.text)
    
    # Additional analysis based on request
    for analysis_type in request.analysis_types:
        try:
            if analysis_type == "summary" and summarizer:
                summary = summarizer(
                    request.text[:1024], 
                    max_length=150, 
                    min_length=50, 
                    do_sample=False
                )
                analysis_results["summary"] = summary[0]['summary_text']
            
            elif analysis_type == "entities" and ner_pipeline:
                entities = ner_pipeline(request.text[:512])
                analysis_results["entities"] = [
                    {
                        "text": ent['word'],
                        "label": ent['entity_group'],
                        "confidence": round(ent['score'], 3)
                    }
                    for ent in entities if ent['score'] > 0.8
                ]
            
            elif analysis_type == "classification":
                # Use Hugging Face API for classification
                hf_result = await analyze_with_huggingface(request.text, "classification")
                analysis_results["classification"] = hf_result
                
        except Exception as e:
            analysis_results[f"{analysis_type}_error"] = str(e)
    
    return {
        "status": "success",
        "text_length": len(request.text),
        "analysis": analysis_results
    }

@app.post("/api/v1/legal-research")
async def legal_research(request: LegalResearchRequest):
    """AI-powered legal research"""
    
    # Search for relevant documents
    search_results = hybrid_search(request.legal_question, 15)
    
    # Analyze the question
    question_analysis = analyze_legal_concepts(request.legal_question)
    
    # Group results by legal area
    results_by_area = defaultdict(list)
    for result in search_results:
        doc_analysis = analyze_legal_concepts(result['text'][:500])
        primary_area = max(
            doc_analysis['legal_areas'].items(),
            key=lambda x: x[1]['confidence'],
            default=('general', {'confidence': 0})
        )[0]
        results_by_area[primary_area].append(result)
    
    # Generate research summary
    research_summary = {
        "question": request.legal_question,
        "legal_areas_identified": list(question_analysis['legal_areas'].keys()),
        "total_documents_found": len(search_results),
        "results_by_area": dict(results_by_area),
        "key_cases": [r for r in search_results[:5] if 'case' in r.get('metadata', {}).get('type', '')],
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
    """Enhanced statistics with AI model info"""
    
    jurisdictions = Counter()
    doc_types = Counter()
    areas = Counter()
    
    for doc_id, metadata in metadata_index.items():
        jurisdictions[metadata.get('jurisdiction', 'unknown')] += 1
        doc_types[metadata.get('type', 'unknown')] += 1
        areas[metadata.get('area', 'unknown')] += 1
    
    return {
        "corpus_info": {
            "total_documents": len(legal_corpus),
            "total_keywords": len(keyword_index),
            "embeddings_available": legal_embeddings is not None,
            "embedding_dimensions": legal_embeddings.shape[1] if legal_embeddings is not None else 0
        },
        "ai_models": {
            "semantic_search": embeddings_model is not None,
            "summarization": summarizer is not None,
            "classification": legal_classifier is not None,
            "named_entity_recognition": ner_pipeline is not None
        },
        "corpus_breakdown": {
            "jurisdictions": dict(jurisdictions.most_common()),
            "document_types": dict(doc_types.most_common()),
            "legal_areas": dict(areas.most_common())
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)