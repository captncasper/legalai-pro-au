"""Australian Legal LLM API - Integrate fine-tuned model with search"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.search import LegalSearchEngine

# Initialize FastAPI
app = FastAPI(
    title="Australian Legal AI - LLM + Search",
    description="Fine-tuned Australian Legal LLM with semantic search",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
llm_model = None
tokenizer = None
search_engine = None
device = None

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    use_search: bool = True
    max_length: int = 500
    temperature: float = 0.7

class SearchRequest(BaseModel):
    query: str
    num_results: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    model_used: str = "australian-legal-llm"

# Startup event
@app.on_event("startup")
async def startup_event():
    global llm_model, tokenizer, search_engine, device
    
    print("🚀 Initializing Australian Legal AI...")
    
    # Initialize search engine
    try:
        search_engine = LegalSearchEngine()
        print("✅ Search engine loaded")
    except Exception as e:
        print(f"⚠️ Search engine initialization failed: {e}")
    
    # Initialize LLM
    model_path = "models/open-australian-legal-llm"
    
    if os.path.exists(model_path):
        try:
            print(f"📚 Loading Australian Legal LLM from {model_path}")
            
            # Detect device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🖥️ Using device: {device}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            if device.type == "cpu":
                llm_model = llm_model.to(device)
            
            print("✅ Australian Legal LLM loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            print("📝 Falling back to search-only mode")
    else:
        print(f"⚠️ Model not found at {model_path}")
        print("📝 Running in search-only mode")

# Chat endpoint - the main attraction!
@app.post("/chat", response_model=ChatResponse)
async def chat_with_legal_ai(request: ChatRequest):
    """
    Chat with the Australian Legal AI
    - Uses fine-tuned LLM for legal expertise
    - Optionally grounds responses with search results
    """
    
    if not llm_model:
        raise HTTPException(
            status_code=503, 
            detail="LLM not available. Please use /search endpoint instead."
        )
    
    try:
        # Step 1: Search for relevant context if requested
        sources = []
        context = ""
        
        if request.use_search and search_engine:
            search_results = search_engine.search(request.query, k=3)
            
            for result in search_results:
                sources.append({
                    'excerpt': result['document'][:200] + "...",
                    'relevance': result['relevance_score'],
                    'jurisdiction': result.get('jurisdiction', ''),
                    'type': result.get('doc_type', '')
                })
                context += f"\n{result['document']}\n"
        
        # Step 2: Prepare prompt
        if context:
            prompt = f"""You are an Australian Legal AI Assistant. Use the following legal documents as context to answer the question accurately.

Context:
{context}

Question: {request.query}

Provide a clear, accurate answer based on Australian law. If citing specific acts or cases, mention them."""
        else:
            prompt = f"""You are an Australian Legal AI Assistant trained on Australian law. 

Question: {request.query}

Provide a clear, accurate answer based on Australian law. If citing specific acts or cases, mention them."""
        
        # Step 3: Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer (remove the prompt)
        answer = response.split("Provide a clear, accurate answer")[1].strip()
        
        return ChatResponse(
            answer=answer,
            sources=sources if sources else None,
            model_used="australian-legal-llm-finetuned"
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Search endpoint (keep your existing functionality)
@app.post("/search")
async def search_legal_documents(request: SearchRequest):
    """Traditional semantic search endpoint"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    results = search_engine.search(request.query, request.num_results)
    return {"query": request.query, "results": results}

# Hybrid endpoint - best of both worlds
@app.post("/analyze")
async def analyze_legal_question(request: ChatRequest):
    """
    Comprehensive legal analysis:
    1. Search for relevant documents
    2. Use LLM to analyze and synthesize
    3. Provide structured legal analysis
    """
    
    # Search for relevant documents
    search_results = []
    if search_engine:
        search_results = search_engine.search(request.query, k=5)
    
    # If LLM available, provide analysis
    if llm_model:
        # Create structured analysis prompt
        context = "\n\n".join([r['document'] for r in search_results[:3]])
        
        analysis_prompt = f"""As an Australian Legal AI, provide a structured analysis of the following question.

Legal Context:
{context}

Question: {request.query}

Provide your analysis in this format:
1. LEGAL ISSUE: Identify the key legal issue(s)
2. RELEVANT LAW: Cite relevant acts, sections, or cases
3. APPLICATION: Apply the law to the question
4. CONCLUSION: Provide a clear conclusion
5. IMPORTANT NOTE: Any limitations or need for professional advice"""

        # Generate analysis
        inputs = tokenizer(analysis_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.7,
                do_sample=True
            )
        
        analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)
        analysis = analysis.split("Provide your analysis in this format:")[1].strip()
        
        return {
            "analysis": analysis,
            "supporting_documents": search_results,
            "disclaimer": "This is AI-generated analysis and should not be considered legal advice. Consult a qualified Australian lawyer for specific legal matters."
        }
    else:
        # Fallback to search results only
        return {
            "search_results": search_results,
            "message": "LLM analysis not available. Showing relevant documents only."
        }

@app.get("/")
def root():
    capabilities = []
    
    if llm_model:
        capabilities.append("🤖 Australian Legal LLM Chat")
        capabilities.append("📊 Legal Document Analysis")
    
    if search_engine:
        capabilities.append("🔍 Semantic Legal Search")
    
    return {
        "service": "Australian Legal AI - Powered by Fine-tuned LLM",
        "version": "2.0.0",
        "capabilities": capabilities,
        "endpoints": {
            "/chat": "Chat with Australian Legal AI (LLM)",
            "/search": "Search legal documents",
            "/analyze": "Comprehensive legal analysis",
            "/docs": "API documentation"
        },
        "status": {
            "llm": "ready" if llm_model else "not loaded",
            "search": "ready" if search_engine else "not loaded"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "llm_loaded": llm_model is not None,
        "search_ready": search_engine is not None,
        "device": str(device) if device else "cpu"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)