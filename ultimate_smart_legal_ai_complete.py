import numpy as np
#!/usr/bin/env python3
"""
ULTIMATE SMART Legal AI - Complete Edition
- Original search + RAG
- Smart AI predictions  
- Document Generation
- Settlement Calculator
- Legal Reasoning Engine
- Timeline Analysis
- Evidence Analysis
"""

from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
from typing import List, Dict, Optional
import pickle
import re
from collections import Counter
import uvicorn
from legal_rag import LegalRAG
from datetime import datetime, timedelta
import json

app = FastAPI(
    title="Ultimate SMART Australian Legal AI",
    description="🚀 The most intelligent legal AI system - Complete Edition",
    version="6.0-COMPLETE"
)

# Load your original search index
with open('data/simple_index.pkl', 'rb') as f:
    search_data = pickle.load(f)
    documents = search_data['documents']

# Initialize RAG
rag_engine = LegalRAG()

# ============= REQUEST MODELS =============
class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

class PredictRequest(BaseModel):
    case_details: str

class RAGRequest(BaseModel):
    question: str
    n_sources: int = 5

class SmartAnalysisRequest(BaseModel):
    case_details: str
    include_documents: bool = False
    salary: Optional[float] = None
    years_service: Optional[int] = None

class TimelineRequest(BaseModel):
    dismissal_date: str  # YYYY-MM-DD format
    claim_type: Optional[str] = "unfair_dismissal"

class EvidenceRequest(BaseModel):
    evidence_items: List[str]
    case_type: str = "unfair_dismissal"

# ============= ORIGINAL FEATURES =============
def keyword_search(query: str, n_results: int = 5) -> List[Dict]:
    """Your original keyword search"""
    words = re.findall(r'\w+', query.lower())
    doc_scores = Counter()
    
    for word in words:
        if word in search_data['keyword_index']:
            for doc_id in search_data['keyword_index'][word]:
                doc_scores[doc_id] += 1
    
    results = []
    for doc_id, score in doc_scores.most_common(n_results):
        doc = documents[doc_id]
        results.append({
            'text': doc['text'][:500] + '...',
            'score': score,
            'citation': doc.get('metadata', {}).get('citation', 'Unknown'),
            'method': 'keyword_search'
        })
    return results

def predict_outcome(case_details: str) -> Dict:
    """Smart case outcome prediction"""
    case_lower = case_details.lower()
    score = 50  # Base score
    factors = []
    
    # Positive indicators
    if 'no warning' in case_lower:
        score += 20
        factors.append("✓ No warnings given (+20%)")
    if 'long service' in case_lower or re.search(r'\d+\s*years', case_lower):
        score += 15
        factors.append("✓ Long service (+15%)")
    if 'good performance' in case_lower:
        score += 10
        factors.append("✓ Good performance history (+10%)")
    
    # Negative indicators
    if 'misconduct' in case_lower:
        score -= 30
        factors.append("✗ Misconduct alleged (-30%)")
    if 'small business' in case_lower:
        score -= 10
        factors.append("✗ Small business employer (-10%)")
    
    return {
        'success_probability': min(max(score, 5), 95),
        'factors': factors,
        'recommendation': "Strong case - proceed" if score > 70 else "Moderate case - gather evidence" if score > 40 else "Weak case - consider settlement",
        'method': 'smart_prediction'
    }

# ============= SMART FEATURES =============

class SettlementCalculator:
    @staticmethod
    def calculate(salary: float, years: int, case_strength: int = 70) -> Dict:
        weekly = salary / 52
        
        # Base calculation
        min_weeks = 4
        typical_weeks = 8 + min(years, 10)  # Cap benefit at 10 years
        max_weeks = 26
        
        # Adjust for case strength
        if case_strength > 80:
            typical_weeks = min(typical_weeks * 1.5, max_weeks)
        elif case_strength < 50:
            typical_weeks = max(min_weeks, typical_weeks * 0.7)
            
        return {
            'weekly_pay': round(weekly, 2),
            'minimum': round(weekly * min_weeks, 2),
            'typical': round(weekly * typical_weeks, 2),
            'maximum': round(weekly * max_weeks, 2),
            'tax_free_portion': min(round(weekly * typical_weeks, 2), 11985),
            'confidence': f"{case_strength}% chance of success"
        }

class DocumentGenerator:
    @staticmethod
    def generate_f8c(details: Dict) -> str:
        return f"""FAIR WORK COMMISSION
FORM F8C - UNFAIR DISMISSAL APPLICATION

APPLICANT: {details.get('name', '[Your Name]')}
EMPLOYER: {details.get('employer', '[Employer Name]')}
DISMISSAL DATE: {details.get('dismissal_date', '[Date]')}

CLAIM: The dismissal was harsh, unjust and unreasonable because:
{details.get('reasons', '[Your reasons here]')}

Lodge within 21 days at www.fwc.gov.au
"""

    @staticmethod
    def generate_witness_statement(details: Dict) -> str:
        return f"""WITNESS STATEMENT

I, {details.get('witness_name', '[Name]')}, state:

1. {details.get('statement', '[Witness account]')}

Signed: _______________ Date: {datetime.now().strftime('%d/%m/%Y')}
"""

class LegalReasoningEngine:
    def analyze(self, case_details: str) -> Dict:
        case_lower = case_details.lower()
        
        # Identify claims
        claims = []
        if any(word in case_lower for word in ['dismiss', 'fired', 'terminated']):
            claims.append('unfair_dismissal')
        if any(word in case_lower for word in ['discriminat', 'harass']):
            claims.append('discrimination')
            
        # Score the case
        score = 50
        factors = []
        
        if 'no warning' in case_lower:
            score += 20
            factors.append("✓ No warnings given (+20%)")
        if re.search(r'\d+\s*year', case_lower):
            score += 15
            factors.append("✓ Long service (+15%)")
        if 'misconduct' in case_lower:
            score -= 30
            factors.append("✗ Misconduct alleged (-30%)")
            
        return {
            'claims': claims,
            'success_probability': min(max(score, 5), 95),
            'factors': factors,
            'next_steps': self._get_next_steps(claims, case_lower)
        }
    
    def _get_next_steps(self, claims: List[str], case_text: str) -> List[str]:
        steps = []
        if 'unfair_dismissal' in claims:
            steps.append("⚡ File F8C within 21 days")
        steps.append("📄 Gather all employment documents")
        steps.append("👥 List potential witnesses")
        return steps

class TimelineAnalyzer:
    @staticmethod
    def calculate_deadlines(dismissal_date: str, claim_type: str = "unfair_dismissal") -> Dict:
        try:
            dismissal = datetime.strptime(dismissal_date, '%Y-%m-%d')
        except:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}
        
        today = datetime.now()
        days_since = (today - dismissal).days
        
        deadlines = {
            'unfair_dismissal': {
                'deadline': dismissal + timedelta(days=21),
                'name': 'Unfair Dismissal (F8C)',
                'critical': True
            },
            'general_protections': {
                'deadline': dismissal + timedelta(days=21),
                'name': 'General Protections',
                'critical': True
            },
            'discrimination': {
                'deadline': dismissal + timedelta(days=180),
                'name': 'Discrimination Complaint',
                'critical': False
            },
            'breach_of_contract': {
                'deadline': dismissal + timedelta(days=2190),  # 6 years
                'name': 'Contract Breach',
                'critical': False
            }
        }
        
        result = {
            'dismissal_date': dismissal_date,
            'days_since_dismissal': days_since,
            'deadlines': []
        }
        
        for claim, info in deadlines.items():
            days_remaining = (info['deadline'] - today).days
            status = 'EXPIRED' if days_remaining < 0 else 'URGENT' if days_remaining < 7 else 'OK'
            
            result['deadlines'].append({
                'claim_type': claim,
                'deadline': info['deadline'].strftime('%Y-%m-%d'),
                'days_remaining': max(0, days_remaining),
                'status': status,
                'critical': info['critical']
            })
        
        # Sort by urgency
        result['deadlines'].sort(key=lambda x: x['days_remaining'])
        result['most_urgent'] = result['deadlines'][0] if result['deadlines'] else None
        
        return result

class EvidenceAnalyzer:
    @staticmethod
    def analyze_evidence(evidence_items: List[str], case_type: str = "unfair_dismissal") -> Dict:
        # Critical evidence for different case types
        evidence_requirements = {
            'unfair_dismissal': {
                'critical': ['termination letter', 'employment contract', 'pay slips'],
                'important': ['performance reviews', 'warnings', 'emails', 'policies'],
                'helpful': ['witness statements', 'medical certificates', 'comparator evidence']
            },
            'discrimination': {
                'critical': ['discriminatory comments', 'comparator evidence', 'complaint records'],
                'important': ['witness statements', 'pattern evidence', 'emails'],
                'helpful': ['policies', 'training records', 'previous complaints']
            }
        }
        
        requirements = evidence_requirements.get(case_type, evidence_requirements['unfair_dismissal'])
        
        # Analyze what user has
        evidence_lower = [item.lower() for item in evidence_items]
        
        has_critical = []
        has_important = []
        has_helpful = []
        missing_critical = []
        
        # Check critical evidence
        for item in requirements['critical']:
            found = any(item in evidence for evidence in evidence_lower)
            if found:
                has_critical.append(item)
            else:
                missing_critical.append(item)
        
        # Check important evidence
        for item in requirements['important']:
            if any(item in evidence for evidence in evidence_lower):
                has_important.append(item)
        
        # Check helpful evidence
        for item in requirements['helpful']:
            if any(item in evidence for evidence in evidence_lower):
                has_helpful.append(item)
        
        # Calculate strength score
        critical_score = len(has_critical) / len(requirements['critical']) * 50
        important_score = len(has_important) / len(requirements['important']) * 30
        helpful_score = len(has_helpful) / len(requirements['helpful']) * 20
        total_score = critical_score + important_score + helpful_score
        
        return {
            'evidence_strength': round(total_score),
            'has_critical': has_critical,
            'has_important': has_important,
            'has_helpful': has_helpful,
            'missing_critical': missing_critical,
            'recommendations': [
                f"⚠️ URGENT: Obtain {item}" for item in missing_critical[:3]
            ] + [
                "✓ Good evidence collection" if total_score > 70 else "⚡ Strengthen evidence urgently"
            ],
            'next_steps': [
                "Organize evidence chronologically",
                "Make copies of all documents",
                "Prepare witness list with contact details"
            ]
        }

# Initialize engines
reasoning_engine = LegalReasoningEngine()
settlement_calc = SettlementCalculator()
doc_generator = DocumentGenerator()
timeline_analyzer = TimelineAnalyzer()
evidence_analyzer = EvidenceAnalyzer()

# ============= API ENDPOINTS =============

@app.get("/")
async def root():
    return {
        "message": "🚀 Ultimate SMART Legal AI - Complete Edition!",
        "endpoints": {
            "search": {
                "/search/keyword": "Original keyword search",
                "/search/semantic": "RAG semantic search with citations"
            },
            "ai": {
                "/predict": "Predict case outcome",
                "/analyze": "Complete case analysis",
                "/analyze/smart": "Smart analysis with all features"
            },
            "rag": {
                "/ask": "Ask question with cited sources",
                "/chat": "Legal chat with RAG"
            },
            "tools": {
                "/timeline/check": "Check legal deadlines",
                "/evidence/analyze": "Analyze evidence strength",
                "/calculate/settlement": "Calculate settlement estimate",
                "/generate/document/{type}": "Generate legal documents"
            }
        },
        "stats": {
            "documents": len(documents),
            "rag_chunks": rag_engine.collection.count()
        }
    }

# Original search endpoint
@app.post("/search/keyword")
async def search_keyword(request: SearchRequest):
    """Original keyword-based search"""
    return {
        "query": request.query,
        "results": keyword_search(request.query, request.n_results),
        "method": "keyword"
    }

# RAG search endpoint
@app.post("/search/semantic")
async def search_semantic(request: SearchRequest):
    """Semantic search with RAG"""
    result = rag_engine.query(request.query, request.n_results)
    return {
        "query": request.query,
        "results": result['sources'],
        "method": "semantic_rag"
    }

# Smart prediction endpoint
@app.post("/predict")
async def predict(request: PredictRequest):
    """Predict case outcome"""
    return predict_outcome(request.case_details)

# RAG Q&A endpoint
@app.post("/ask")
async def ask(request: RAGRequest):
    """Ask question and get answer with citations"""
    return rag_engine.query(request.question, request.n_sources)

# Combined analysis endpoint
@app.post("/analyze")
async def analyze(request: PredictRequest):
    """Complete analysis: prediction + search + RAG"""
    case_details = request.case_details
    
    # 1. Predict outcome
    prediction = predict_outcome(case_details)
    
    # 2. Keyword search
    keyword_results = keyword_search(case_details, 3)
    
    # 3. RAG search
    rag_result = rag_engine.query(case_details, 3)
    
    return {
        "case_details": case_details,
        "prediction": prediction,
        "keyword_matches": keyword_results,
        "semantic_sources": rag_result['sources'],
        "rag_answer": rag_result['answer'],
        "recommendations": [
            f"Success probability: {prediction['success_probability']}%",
            f"Found {len(keyword_results)} keyword matches",
            f"Found {len(rag_result['sources'])} semantic matches",
            "Consider cited cases for precedent"
        ]
    }

@app.post("/analyze/smart")
async def smart_analysis(request: SmartAnalysisRequest):
    """Complete smart analysis with all features"""
    
    # 1. Legal reasoning
    reasoning = reasoning_engine.analyze(request.case_details)
    
    # 2. RAG search for precedents
    rag_result = rag_engine.query(request.case_details, 5)
    
    # 3. Settlement calculation if salary provided
    settlement = None
    if request.salary:
        years = request.years_service or 2
        settlement = settlement_calc.calculate(
            request.salary, 
            years, 
            reasoning['success_probability']
        )
    
    # 4. Generate documents if requested
    documents_generated = {}
    if request.include_documents:
        documents_generated['f8c'] = doc_generator.generate_f8c({
            'name': 'Applicant',
            'reasons': request.case_details
        })
    
    return {
        'analysis': reasoning,
        'precedents': rag_result['sources'][:3],
        'settlement_estimate': settlement,
        'documents': documents_generated,
        'executive_summary': {
            'win_chance': f"{reasoning['success_probability']}%",
            'best_claim': reasoning['claims'][0] if reasoning['claims'] else 'general_dispute',
            'potential_payout': settlement['typical'] if settlement else 'Provide salary for estimate',
            'urgent': '🚨 File within 21 days!' if 'unfair_dismissal' in reasoning['claims'] else '📋 No urgent deadline'
        }
    }

# Timeline endpoint
@app.post("/timeline/check")
async def check_timeline(request: TimelineRequest):
    """Check legal deadlines"""
    return timeline_analyzer.calculate_deadlines(request.dismissal_date, request.claim_type)

# Evidence analysis endpoint
@app.post("/evidence/analyze")
async def analyze_evidence(request: EvidenceRequest):
    """Analyze evidence strength"""
    return evidence_analyzer.analyze_evidence(request.evidence_items, request.case_type)

# Document generation
@app.post("/generate/document/{doc_type}")
async def generate_document(doc_type: str, details: Dict):
    """Generate legal documents"""
    if doc_type == 'f8c':
        return {'content': doc_generator.generate_f8c(details)}
    elif doc_type == 'witness':
        return {'content': doc_generator.generate_witness_statement(details)}
    else:
        raise HTTPException(400, "Unknown document type")

# Settlement calculation
@app.post("/calculate/settlement")
async def calculate_settlement(salary: float, years: int, case_strength: int = 70):
    """Calculate settlement with tax implications"""
    return settlement_calc.calculate(salary, years, case_strength)

# Chat endpoint
@app.post("/chat")
async def chat(message: str):
    """Chat interface using RAG"""
    result = rag_engine.query(message)
    
    return {
        "user": message,
        "assistant": result['answer'],
        "sources_used": len(result['sources']),
        "confidence": "high" if result['sources'] else "low"
    }

# WebSocket for real-time chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    await websocket.send_json({
        "message": "Connected to Smart Legal AI Assistant",
        "features": ["reasoning", "documents", "timeline", "evidence", "chat"]
    })
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Quick analysis
            reasoning = reasoning_engine.analyze(data)
            
            response = {
                'analysis': {
                    'claims': reasoning['claims'],
                    'success_rate': reasoning['success_probability'],
                    'urgent': 'unfair_dismissal' in reasoning['claims']
                },
                'suggestion': "Would you like me to check deadlines or analyze your evidence?"
            }
            
            await websocket.send_json(response)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ULTIMATE SMART LEGAL AI - v6.0 COMPLETE")
    print("=" * 60)
    print("✅ Keyword + Semantic search")
    print("✅ Smart legal reasoning") 
    print("✅ Document generation")
    print("✅ Settlement calculator")
    print("✅ Timeline tracking")
    print("✅ Evidence analysis")
    print("✅ WebSocket real-time chat")
    print("=" * 60)
    print("📖 Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
