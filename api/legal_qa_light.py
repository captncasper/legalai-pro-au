"""
Lightweight Australian Legal Q&A API - Works in Codespaces!
No large model downloads - uses search + templates or HF Inference API
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import sys
import uuid
from datetime import datetime
import httpx

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import search engine
try:
    from src.search import LegalSearchEngine
    SEARCH_AVAILABLE = True
except:
    SEARCH_AVAILABLE = False
    print("⚠️ Search engine not available")

# Initialize FastAPI
app = FastAPI(
    title="Australian Legal Q&A API - Lightweight",
    description="AI-powered legal Q&A - $0.10 per question",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize search engine if available
search_engine = None
if SEARCH_AVAILABLE:
    try:
        search_engine = LegalSearchEngine()
        print("✅ Search engine loaded")
    except Exception as e:
        print(f"⚠️ Search engine failed: {e}")

# Simple API key database
api_keys_db = {
    "demo_key": {"name": "Demo User", "credits": 20, "rate": 0.10},
    "test_premium": {"name": "Test Premium", "credits": 1000, "rate": 0.05}
}

# Track usage
usage_db = {}

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    use_search: bool = True
    max_length: int = 300

class AnswerResponse(BaseModel):
    answer: str
    question: str
    sources: Optional[List[Dict]] = None
    cost: float = 0.10
    credits_remaining: Optional[float] = None
    method: str = "template"  # template, search, or ai

# Validate API key
def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    if api_key not in api_keys_db:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    user = api_keys_db[api_key]
    if user["credits"] <= 0:
        raise HTTPException(status_code=402, detail="No credits remaining")
    
    return api_key

# Template-based responses for common questions
def get_template_response(question: str) -> Optional[str]:
    """Return template response for common questions"""
    q_lower = question.lower()
    
    templates = {
        "unfair dismissal": """Under the Fair Work Act 2009, unfair dismissal claims require:
• Minimum employment period (6 months for businesses with 15+ employees, 12 months for smaller)
• The dismissal was harsh, unjust or unreasonable
• Not a case of genuine redundancy
• Must lodge within 21 days of dismissal
• Remedies include reinstatement or compensation (capped at 26 weeks' pay)""",
        
        "director duties": """Directors' duties under the Corporations Act 2001 include:
• Duty of care and diligence (s180) - act with reasonable care
• Duty of good faith (s181) - act in company's best interests
• Proper use of position (s182) - no misuse for personal gain
• Proper use of information (s183) - no misuse of company info
• Civil penalties up to $1.11 million per breach""",
        
        "contract": """Australian contract law requires:
• Offer - clear proposal to enter agreement
• Acceptance - unqualified agreement to offer
• Consideration - something of value exchanged
• Intention to create legal relations
• Capacity - parties must be able to contract
• Genuine consent - no duress, undue influence, or misrepresentation""",
        
        "negligence": """Negligence in Australian law requires proving:
• Duty of care owed to the plaintiff
• Breach of that duty (failure to meet standard of care)
• Causation - breach caused the harm
• Damage - actual loss or injury occurred
• Defences include contributory negligence and voluntary assumption of risk"""
    }
    
    # Find matching template
    for key, response in templates.items():
        if key in q_lower:
            return response
    
    return None

# Generate answer using search + templates
def generate_answer(question: str, use_search: bool = True) -> Dict:
    """Generate answer using available methods"""
    
    # Try template first
    template_answer = get_template_response(question)
    if template_answer:
        return {
            "answer": template_answer,
            "method": "template",
            "sources": None
        }
    
    # Try search if available
    if use_search and search_engine:
        try:
            results = search_engine.search(question, k=3)
            if results:
                # Combine top results
                answer_parts = []
                sources = []
                
                for i, result in enumerate(results[:2]):
                    if result['relevance_score'] > 0.3:
                        doc_excerpt = result['document'][:300]
                        answer_parts.append(doc_excerpt)
                        sources.append({
                            "text": doc_excerpt[:100] + "...",
                            "score": result['relevance_score'],
                            "type": result.get('doc_type', 'Unknown')
                        })
                
                if answer_parts:
                    answer = "Based on Australian legal documents:\n\n" + "\n\n".join(answer_parts)
                    return {
                        "answer": answer,
                        "method": "search",
                        "sources": sources
                    }
        except Exception as e:
            print(f"Search error: {e}")
    
    # Fallback to general response
    return {
        "answer": """I can help with questions about Australian law including:
• Employment law (Fair Work Act, unfair dismissal)
• Corporate law (directors' duties, company structure)
• Contract law (formation, breach, remedies)
• Tort law (negligence, defamation)
• Consumer law (guarantees, misleading conduct)

Please ask a specific question about one of these areas.""",
        "method": "fallback",
        "sources": None
    }

# Option to use HuggingFace Inference API (no download needed)
async def use_hf_inference_api(question: str, api_key: str = None):
    """Use HuggingFace Inference API - no model download needed"""
    
    # You need a HF API key for this
    hf_api_key = os.getenv("HF_API_KEY", api_key)
    if not hf_api_key:
        return None
    
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    
    # Use the model via API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api-inference.huggingface.co/models/umarbutler/open-australian-legal-llm",
            headers=headers,
            json={"inputs": f"Question: {question}\nAnswer:"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text']
    
    return None

# Main Q&A endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    api_key: str = Depends(validate_api_key)
):
    """Ask a legal question - uses search and templates, no large downloads"""
    
    user = api_keys_db[api_key]
    
    # Generate answer
    result = generate_answer(request.question, request.use_search)
    
    # Deduct credits
    user["credits"] -= user["rate"]
    
    # Track usage
    if api_key not in usage_db:
        usage_db[api_key] = []
    usage_db[api_key].append({
        "question": request.question,
        "timestamp": datetime.now().isoformat(),
        "method": result["method"]
    })
    
    return AnswerResponse(
        answer=result["answer"],
        question=request.question,
        sources=result.get("sources"),
        cost=user["rate"],
        credits_remaining=user["credits"],
        method=result["method"]
    )

# Simple search endpoint
@app.post("/search")
async def search_documents(
    query: str,
    num_results: int = 5
):
    """Search legal documents"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search not available")
    
    results = search_engine.search(query, num_results)
    return {"query": query, "results": results}

# Web interface
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Australian Legal Q&A</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f0f5f9; padding: 30px; border-radius: 10px; }
            h1 { color: #1a5490; }
            textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #1a5490; color: white; padding: 12px 30px; border: none; 
                    border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #2a6bb0; }
            .answer { background: white; padding: 20px; margin-top: 20px; border-radius: 5px; 
                     box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .sources { background: #f8f9fa; padding: 10px; margin-top: 10px; border-radius: 5px; }
            .price { color: #e74c3c; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦘 Australian Legal Q&A Service</h1>
            <p>Get instant answers about Australian law - <span class="price">$0.10 per question</span></p>
            
            <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <strong>Demo:</strong> You have 20 free questions. API key: demo_key
            </div>
            
            <h3>Ask Your Question:</h3>
            <textarea id="question" rows="3" placeholder="e.g. What are the requirements for unfair dismissal?"></textarea>
            
            <br><br>
            <button onclick="askQuestion()">Get Answer</button>
            
            <div id="loading" style="display:none; margin-top:20px;">⏳ Searching Australian law...</div>
            
            <div id="answer" class="answer" style="display:none;">
                <h3>Answer:</h3>
                <div id="answer-text"></div>
                <div id="sources" class="sources" style="display:none;"></div>
                <hr>
                <small><em>This is general information only. Consult a lawyer for specific advice.</em></small>
            </div>
        </div>
        
        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value;
                if (!question) return;
                
                document.getElementById('answer').style.display = 'none';
                document.getElementById('loading').style.display = 'block';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer demo_key'
                        },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const data = await response.json();
                    
                    document.getElementById('loading').style.display = 'none';
                    
                    if (response.ok) {
                        document.getElementById('answer-text').innerText = data.answer;
                        
                        if (data.sources && data.sources.length > 0) {
                            let sourcesHtml = '<strong>Sources:</strong><br>';
                            data.sources.forEach((s, i) => {
                                sourcesHtml += `${i+1}. ${s.text} (relevance: ${s.score.toFixed(2)})<br>`;
                            });
                            document.getElementById('sources').innerHTML = sourcesHtml;
                            document.getElementById('sources').style.display = 'block';
                        } else {
                            document.getElementById('sources').style.display = 'none';
                        }
                        
                        document.getElementById('answer').style.display = 'block';
                    } else {
                        alert('Error: ' + data.detail);
                    }
                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    alert('Error: Could not connect to server');
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "search_available": search_engine is not None,
        "methods": ["template", "search", "fallback"]
    }

@app.get("/pricing")
async def pricing():
    return {
        "pricing": {
            "demo": "20 free questions with demo_key",
            "pay_as_you_go": "$0.10 per question",
            "starter": "$50 for 600 questions",
            "professional": "$200 for 3000 questions"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting lightweight Q&A server...")
    print("✅ No large model downloads needed!")
    print("📊 Using search + smart templates")
    uvicorn.run(app, host="0.0.0.0", port=8000)