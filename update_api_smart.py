#!/usr/bin/env python3
"""Update your API to use smart features"""

print('''
Add this to your legal_ai_supreme_au.py:

from add_semantic_search import SemanticSearchEngine
from train_outcome_predictor import OutcomePredictor
import pickle

# Initialize smart components
semantic_engine = SemanticSearchEngine()
with open('outcome_predictor.pkl', 'rb') as f:
    outcome_model = pickle.load(f)

@app.post("/api/v1/search/semantic")
async def semantic_search(request: dict):
    query = request.get("query", "")
    results = semantic_engine.semantic_search(query, top_k=10)
    
    formatted_results = []
    for r in results:
        formatted_results.append({
            "citation": r['case']['citation'],
            "case_name": r['case']['case_name'],
            "similarity_score": r['similarity_score'],
            "outcome": r['case']['outcome'],
            "snippet": r['case']['text'][:200] + "..."
        })
    
    return {
        "query": query,
        "results": formatted_results,
        "search_type": "semantic"
    }

@app.post("/api/v1/predict/outcome")
async def predict_outcome(request: dict):
    description = request.get("description", "")
    
    # Get prediction
    predictor = OutcomePredictor()
    prediction = predictor.predict_new_case(description)
    
    # Find similar cases
    similar_cases = semantic_engine.semantic_search(description, top_k=5)
    
    return {
        "prediction": prediction,
        "similar_cases": [
            {
                "citation": c['case']['citation'],
                "outcome": c['case']['outcome'],
                "similarity": c['similarity_score']
            } for c in similar_cases
        ],
        "confidence": max(prediction['applicant_wins'], 
                         prediction['settles'], 
                         prediction['applicant_loses'])
    }
''')
