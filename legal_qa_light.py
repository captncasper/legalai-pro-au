import numpy as np
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
import pickle
import re
from collections import Counter

# Load search index
try:
    with open('data/simple_index.pkl', 'rb') as f:
        search_data = pickle.load(f)
    print(f"Loaded search index: {len(search_data['documents'])} documents")
except Exception as e:
    print(f"Warning: Could not load search index: {e}")
    search_data = None

def real_search(query, num_results=5):
    """Search using the actual index"""
    if not search_data:
        return []
    
    words = re.findall(r'\w+', query.lower())
    doc_scores = Counter()
    
    # Score documents
    for word in words:
        if word in search_data['keyword_index']:
            for doc_id in search_data['keyword_index'][word]:
                doc_scores[doc_id] += 1
    
    # Get top results
    results = []
    for doc_id, score in doc_scores.most_common(num_results):
        doc = search_data['documents'][doc_id]
        results.append({
            'text': doc['text'],
            'snippet': doc['text'][:300] + '...',
            'score': score,
            'metadata': doc.get('metadata', {}),
            'citation': doc.get('metadata', {}).get('citation', 'Australian Legal Document')
        })
    
    return results
import os
import sys
import uuid
from datetime import datetime
import httpx
import re

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
        "unfair dismissal": """**Overview:**
Unfair dismissal protections under the Fair Work Act 2009 (Cth) protect employees from being dismissed in a manner that is harsh, unjust or unreasonable.

**Eligibility Requirements:**
1. **Minimum Employment Period:**
   • 6 months for businesses with 15+ employees
   • 12 months for smaller businesses
   
2. **Employment Type:**
   • Must be an employee (not contractor)
   • Covered by the national workplace relations system

**Key Elements for a Claim:**
• The dismissal was harsh, unjust or unreasonable
• Procedural fairness was not followed
• Not a case of genuine redundancy
• Not excluded for other reasons (e.g., high income threshold)

**Time Limits:**
• Must lodge application within 21 days of dismissal
• Extensions only granted in exceptional circumstances

**Potential Remedies:**
• Reinstatement to former position
• Compensation (capped at 26 weeks' pay)
• Maximum compensation also subject to compensation cap ($87,500 as of 2024)""",
        
        "director duties": """**Overview:**
Directors owe statutory and fiduciary duties to their company under the Corporations Act 2001 (Cth) and common law.

**Statutory Duties (Corporations Act 2001):**
1. **Duty of Care and Diligence (s180):**
   • Act with reasonable care and diligence
   • Business judgment rule provides protection

2. **Good Faith and Proper Purpose (s181):**
   • Act in good faith in company's best interests
   • Exercise powers for proper purposes

3. **Use of Position (s182):**
   • Not improperly use position to gain advantage
   • Applies to former directors as well

4. **Use of Information (s183):**
   • Not misuse company information
   • Continues after ceasing to be director

**Penalties:**
• Civil penalties up to $1.11 million per breach
• Criminal sanctions for dishonest conduct
• Personal liability for company debts in some cases
• Disqualification from managing corporations""",
        
        "contract": """**Overview:**
A valid contract under Australian law requires several essential elements to be legally enforceable.

**Essential Elements:**
1. **Offer:**
   • Clear and unequivocal proposal
   • Must be communicated to offeree
   • Can be revoked before acceptance

2. **Acceptance:**
   • Unqualified agreement to all terms
   • Must be communicated (silence ≠ acceptance)
   • Mirror image rule applies

3. **Consideration:**
   • Something of value exchanged
   • Must be sufficient but need not be adequate
   • Past consideration is not valid

4. **Intention to Create Legal Relations:**
   • Presumed in commercial contexts
   • Presumed against in domestic/social contexts

5. **Capacity:**
   • Parties must have legal capacity
   • Minors, mental incapacity, intoxication

6. **Genuine Consent:**
   • No duress, undue influence, misrepresentation
   • No mistake or unconscionable conduct""",
        
        "negligence": """**Overview:**
Negligence is a tort requiring proof of four essential elements to establish liability.

**Essential Elements:**
1. **Duty of Care:**
   • Reasonable foreseeability of harm
   • Proximity/neighbourhood principle
   • Policy considerations

2. **Breach of Duty:**
   • Failure to meet standard of reasonable person
   • Magnitude of risk vs burden of precautions
   • Professional standard for skilled defendants

3. **Causation:**
   • Factual causation ('but for' test)
   • Legal causation (scope of liability)
   • No intervening acts breaking chain

4. **Damage:**
   • Actual loss or injury suffered
   • Must not be too remote
   • Includes personal injury, property damage, economic loss

**Defences:**
• Contributory negligence (reduces damages)
• Voluntary assumption of risk
• Illegality
• Statutory limitations (Civil Liability Acts)"""
    }
    
    # Find matching template
    for key, response in templates.items():
        if key in q_lower:
            return response
    
    return None

# Generate answer using search + templates
def generate_answer(question: str, use_search: bool = True) -> Dict:
    """Generate answer using available methods"""
    
    # Extract key terms from question
    q_lower = question.lower()
    
    # Try template first for exact matches
    template_answer = get_template_response(question)
    if template_answer:
        # Structure the template answer better
        structured_answer = structure_legal_answer(question, template_answer, "legislation")
        return {
            "answer": structured_answer,
            "method": "template",
            "sources": None
        }
    
    # Try search if available
    if use_search and search_engine:
        try:
            results = search_engine.search(question, k=5)  # Get more results
            if results:
                # Filter relevant results based on question keywords
                relevant_results = filter_relevant_results(question, results)
                
                if relevant_results:
                    # Create structured answer from search results
                    answer = create_structured_answer(question, relevant_results)
                    
                    # Create proper sources
                    sources = []
                    for result in relevant_results[:3]:
                        sources.append({
                            "text": result['document'][:150] + "...",
                            "score": result['relevance_score'],
                            "type": result.get('doc_type', 'Unknown'),
                            "jurisdiction": result.get('jurisdiction', 'Commonwealth')
                        })
                    
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

def filter_relevant_results(question: str, results: List[Dict]) -> List[Dict]:
    """Filter search results to only include relevant ones"""
    q_lower = question.lower()
    
    # Define topic keywords
    topic_keywords = {
        "unfair dismissal": ["dismissal", "fair work", "employment", "termination", "employee"],
        "contract": ["contract", "agreement", "offer", "acceptance", "consideration"],
        "negligence": ["negligence", "duty of care", "breach", "damage", "tort"],
        "director": ["director", "duties", "corporation", "fiduciary", "company"],
        "consumer": ["consumer", "acl", "guarantee", "refund", "warranty"]
    }
    
    # Find the topic
    topic = None
    for topic_name, keywords in topic_keywords.items():
        if any(keyword in q_lower for keyword in keywords):
            topic = topic_name
            break
    
    if not topic:
        # If no specific topic, return top results with good scores
        return [r for r in results if r['relevance_score'] > 0.5]
    
    # Filter results based on topic keywords
    filtered = []
    topic_words = topic_keywords.get(topic, [])
    
    for result in results:
        doc_lower = result['document'].lower()
        # Check if document contains topic keywords
        if any(keyword in doc_lower for keyword in topic_words):
            filtered.append(result)
    
    # If no filtered results, return high-scoring ones
    if not filtered:
        filtered = [r for r in results if r['relevance_score'] > 0.6]
    
    return filtered[:3]  # Return top 3 relevant results

def create_structured_answer(question: str, results: List[Dict]) -> str:
    """Create a well-structured answer from search results"""
    
    # Determine the type of question
    q_lower = question.lower()
    
    # Start with a clear introduction
    if "what" in q_lower or "explain" in q_lower:
        answer = "Based on Australian law:\n\n"
    elif "how" in q_lower:
        answer = "The process under Australian law:\n\n"
    elif "when" in q_lower:
        answer = "According to Australian law:\n\n"
    else:
        answer = "Under Australian law:\n\n"
    
    # Extract key information from results
    key_points = []
    legislation_mentioned = set()
    
    for result in results:
        doc = result['document']
        
        # Extract legislation references
        import re
        acts = re.findall(r'[A-Z][^\.]*Act \d{4}', doc)
        legislation_mentioned.update(acts)
        
        # Extract key sentences (look for sentences with important keywords)
        sentences = doc.split('. ')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['require', 'must', 'include', 'means', 'defined']):
                if len(sentence) > 30:  # Avoid fragments
                    key_points.append(sentence.strip())
    
    # Structure the answer
    if key_points:
        # Add main points
        answer += "**Key Requirements:**\n"
        for i, point in enumerate(key_points[:5], 1):
            answer += f"{i}. {point}\n"
    
    # Add relevant legislation
    if legislation_mentioned:
        answer += "\n**Relevant Legislation:**\n"
        for act in sorted(legislation_mentioned)[:3]:
            answer += f"• {act}\n"
    
    # Add a summary if we have enough content
    if len(results) > 1:
        answer += "\n**Summary:**\n"
        # Use the highest scoring result for summary
        summary = results[0]['document'][:200]
        answer += summary + "..."
    
    return answer

def structure_legal_answer(question: str, content: str, answer_type: str) -> str:
    """Structure any answer in a professional legal format"""
    
    structured = f"**Legal Position on {question}**\n\n"
    
    # Add the main content
    structured += "**Requirements:**\n"
    structured += content + "\n\n"
    
    # Add standard elements based on type
    if answer_type == "legislation":
        structured += "**Important Notes:**\n"
        structured += "• Time limits may apply\n"
        structured += "• Seek legal advice for your specific situation\n"
        structured += "• This is general information only\n"
    
    return structured

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
            .answer strong { color: #1a5490; display: block; margin-top: 15px; margin-bottom: 5px; }
            .answer ul { margin: 5px 0; }
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
                        // Convert markdown-style formatting to HTML
                        let formattedAnswer = data.answer
                            .replace('/**'*\*(.*?)\*\***/g', '<strong>$1</strong>')
                            .replace('/**'n**/g', '<br>')
                            .replace('/**'• **/g', '&bull; ');
                        
                        document.getElementById('answer-text').innerHTML = formattedAnswer;
                        
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