import numpy as np
"""
Australian Legal Q&A Service - Start Making Money Today!
Uses HuggingFace model - no GPU needed!
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import time
import uuid
from datetime import datetime
from transformers import pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Initialize FastAPI
app = FastAPI(
    title="Australian Legal Q&A API",
    description="AI-powered legal Q&A for Australian law - $0.10 per question",
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

# Global model (loads once)
print("🚀 Loading Australian Legal AI model...")
legal_ai = None
executor = ThreadPoolExecutor(max_workers=3)

# Simple in-memory usage tracking (use database in production)
usage_db = {}
api_keys_db = {
    "demo_key": {"name": "Demo User", "credits": 10, "rate": 0.10},
    "test_key_premium": {"name": "Test Premium", "credits": 1000, "rate": 0.05}
}

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None
    max_length: int = 300
    include_sources: bool = True

class AnswerResponse(BaseModel):
    answer: str
    question: str
    session_id: str
    sources: Optional[List[str]] = None
    disclaimer: str = "This is AI-generated information about Australian law. Always consult a qualified lawyer for legal advice."
    cost: float = 0.10
    credits_remaining: Optional[float] = None

class UsageStats(BaseModel):
    total_questions: int
    total_cost: float
    credits_remaining: float
    questions_today: int

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    global legal_ai
    try:
        legal_ai = pipeline(
            'text-generation', 
            model='umarbutler/open-australian-legal-llm',
            device=-1  # CPU
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Fallback to a smaller model if needed
        legal_ai = None

# API key validation
def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    if api_key not in api_keys_db:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    user = api_keys_db[api_key]
    if user["credits"] <= 0:
        raise HTTPException(status_code=402, detail="No credits remaining. Please add credits.")
    
    return api_key

# Helper function to generate answer
def generate_legal_answer(question: str, context: str = None, max_length: int = 300):
    """Generate answer using the legal AI model"""
    
    # Build prompt
    if context:
        prompt = f"""You are an expert in Australian law. Answer based on the context provided.

Context: {context}

Question: {question}

Answer: Based on Australian law,"""
    else:
        prompt = f"""You are an expert in Australian law. Provide accurate information about Australian legislation, case law, and legal principles.

Question: {question}

Answer: Based on Australian law,"""
    
    if legal_ai:
        # Generate with the model
        result = legal_ai(prompt, max_length=max_length, temperature=0.7, do_sample=True)
        answer = result[0]['generated_text'].split("Answer: Based on Australian law,")[-1].strip()
    else:
        # Fallback responses if model not loaded
        answer = "The Australian legal system includes federal and state laws. For specific advice about your situation, please consult a qualified legal professional."
    
    return answer

# Main Q&A endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask_legal_question(
    request: QuestionRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Ask a question about Australian law
    - Costs $0.10 per question (demo key gets 10 free questions)
    - Returns AI-generated answer with sources
    """
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Track usage
    user = api_keys_db[api_key]
    cost = user["rate"]
    
    # Generate answer asynchronously
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        executor,
        generate_legal_answer,
        request.question,
        request.context,
        request.max_length
    )
    
    # Deduct credits
    user["credits"] -= cost
    
    # Track usage
    if api_key not in usage_db:
        usage_db[api_key] = []
    usage_db[api_key].append({
        "question": request.question,
        "timestamp": datetime.now().isoformat(),
        "cost": cost
    })
    
    # Generate sources (in production, these would come from your search engine)
    sources = None
    if request.include_sources:
        sources = [
            "Fair Work Act 2009 (Cth)",
            "Corporations Act 2001 (Cth)",
            "Australian Consumer Law"
        ]
    
    return AnswerResponse(
        answer=answer,
        question=request.question,
        session_id=session_id,
        sources=sources,
        cost=cost,
        credits_remaining=user["credits"]
    )

# Batch questions endpoint (for efficiency)
@app.post("/ask-batch")
async def ask_multiple_questions(
    questions: List[str],
    api_key: str = Depends(validate_api_key)
):
    """Ask multiple questions at once - 20% discount for batches of 5+"""
    
    user = api_keys_db[api_key]
    base_rate = user["rate"]
    
    # Apply discount for batches
    if len(questions) >= 5:
        rate = base_rate * 0.8  # 20% discount
    else:
        rate = base_rate
    
    total_cost = len(questions) * rate
    
    if user["credits"] < total_cost:
        raise HTTPException(status_code=402, detail=f"Insufficient credits. Need {total_cost}, have {user['credits']}")
    
    # Process questions
    answers = []
    for question in questions:
        answer = await loop.run_in_executor(
            executor,
            generate_legal_answer,
            question,
            None,
            200  # Shorter answers for batch
        )
        answers.append({
            "question": question,
            "answer": answer
        })
    
    # Deduct credits
    user["credits"] -= total_cost
    
    return {
        "answers": answers,
        "total_cost": total_cost,
        "credits_remaining": user["credits"],
        "discount_applied": len(questions) >= 5
    }

# Usage statistics endpoint
@app.get("/usage", response_model=UsageStats)
async def get_usage_stats(api_key: str = Depends(validate_api_key)):
    """Get your usage statistics and remaining credits"""
    
    user = api_keys_db[api_key]
    user_usage = usage_db.get(api_key, [])
    
    # Calculate stats
    total_questions = len(user_usage)
    total_cost = sum(u["cost"] for u in user_usage)
    
    # Questions today
    today = datetime.now().date()
    questions_today = sum(
        1 for u in user_usage 
        if datetime.fromisoformat(u["timestamp"]).date() == today
    )
    
    return UsageStats(
        total_questions=total_questions,
        total_cost=total_cost,
        credits_remaining=user["credits"],
        questions_today=questions_today
    )

# Simple web interface
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Australian Legal Q&A - AI Powered</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 30px; border-radius: 10px; }
            h1 { color: #2c3e50; }
            .question-box { width: 100%; padding: 10px; font-size: 16px; }
            .answer-box { background: white; padding: 20px; margin-top: 20px; border-radius: 5px; }
            .button { background: #3498db; color: white; padding: 12px 30px; border: none; 
                     border-radius: 5px; font-size: 16px; cursor: pointer; }
            .button:hover { background: #2980b9; }
            .price { color: #e74c3c; font-weight: bold; }
            .sources { color: #7f8c8d; font-size: 14px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦘 Australian Legal Q&A Service</h1>
            <p>Get instant answers to Australian legal questions - <span class="price">$0.10 per question</span></p>
            
            <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <strong>Demo Mode:</strong> You have 10 free questions. Get your API key for unlimited access.
            </div>
            
            <h3>Ask Your Legal Question:</h3>
            <textarea class="question-box" id="question" rows="3" 
                placeholder="e.g., What are the requirements for unfair dismissal in Australia?"></textarea>
            
            <br><br>
            <button class="button" onclick="askQuestion()">Ask Question ($0.10)</button>
            
            <div id="answer" class="answer-box" style="display:none;">
                <h3>Answer:</h3>
                <p id="answer-text"></p>
                <div class="sources" id="sources"></div>
                <hr>
                <small><em>This is AI-generated information. Always consult a qualified lawyer for legal advice.</em></small>
            </div>
        </div>
        
        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value;
                if (!question) return;
                
                document.getElementById('answer').style.display = 'none';
                
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
                    
                    if (response.ok) {
                        document.getElementById('answer-text').textContent = data.answer;
                        if (data.sources) {
                            document.getElementById('sources').innerHTML = 
                                '<strong>Sources:</strong> ' + data.sources.join(', ');
                        }
                        document.getElementById('answer').style.display = 'block';
                    } else {
                        alert('Error: ' + data.detail);
                    }
                } catch (error) {
                    alert('Error connecting to server');
                }
            }
        </script>
    </body>
    </html>
    """
    return html

# API documentation additions
@app.get("/pricing")
async def pricing_info():
    """Get pricing information"""
    return {
        "pricing": {
            "pay_as_you_go": "$0.10 per question",
            "starter_pack": "$50 for 600 questions ($0.083 each)",
            "professional": "$200 for 3000 questions ($0.067 each)",
            "enterprise": "$500 for 10000 questions ($0.05 each)",
            "batch_discount": "20% off for batches of 5+ questions"
        },
        "features": [
            "Australian law expertise",
            "Instant responses",
            "Source citations",
            "API access",
            "Batch processing",
            "Usage statistics"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": legal_ai is not None,
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)