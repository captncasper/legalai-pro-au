import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
from collections import Counter
import uvicorn
from datetime import datetime

app = FastAPI(title="Ultimate Legal AI v2 - ALL Features")

# Load your search index
with open('data/simple_index.pkl', 'rb') as f:
    search_data = pickle.load(f)
    documents = search_data['documents']

class Request(BaseModel):
    text: str

# Combine all your features
@app.post("/do-everything")
async def do_everything(request: Request):
    """One endpoint that does EVERYTHING"""
    text = request.text.lower()
    
    # 1. Search relevant docs
    words = re.findall(r'\w+', text)
    doc_scores = Counter()
    for word in words:
        if word in search_data['keyword_index']:
            for doc_id in search_data['keyword_index'][word]:
                doc_scores[doc_id] += 1
    
    top_docs = []
    for doc_id, score in doc_scores.most_common(3):
        top_docs.append(documents[doc_id]['text'][:200])
    
    # 2. Predict outcome
    score = 50
    if 'no warning' in text: score += 20
    if 'long service' in text: score += 15
    if 'good performance' in text: score += 10
    if 'small business' in text: score -= 10
    if 'misconduct' in text: score -= 30
    
    # 3. Generate strategy
    if score > 70:
        strategy = "File immediately - strong case!"
    elif score > 40:
        strategy = "Gather more evidence first"
    else:
        strategy = "Try negotiation - weak case"
    
    # 4. Generate letter
    letter = f"""Dear Sir/Madam,
    
I write regarding my dismissal. {text}

This was unfair because:
1. No proper process followed
2. No opportunity to respond
3. Disproportionate to alleged conduct

I seek reinstatement or compensation.

Yours faithfully,
[Name]"""
    
    # 5. Timeline check
    if '21 days' in text or 'deadline' in text:
        timeline = "⚠️ URGENT: 21 day deadline for unfair dismissal!"
    else:
        timeline = "Check your deadlines"
    
    return {
        "success_probability": f"{score}%",
        "strategy": strategy,
        "relevant_cases": top_docs,
        "letter_draft": letter,
        "timeline_warning": timeline,
        "next_steps": [
            "1. File within 21 days",
            "2. Gather all documents", 
            "3. List witnesses",
            "4. Calculate losses"
        ]
    }

@app.get("/")
async def root():
    return {
        "message": "Ultimate Legal AI v2 - Everything in ONE endpoint!",
        "usage": "POST /do-everything with {\"text\": \"your case details\"}"
    }

if __name__ == "__main__":
    print("🚀 Ultimate Legal AI v2 - EVERYTHING in one API!")
    uvicorn.run(app, host="0.0.0.0", port=8003)
