import numpy as np
#!/usr/bin/env python3
"""
Enhanced Legal API with Corpus Intelligence
Uses patterns learned from your corpus
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import pickle
import re
from legal_rag import LegalRAG
import uvicorn

app = FastAPI(
    title="Australian Legal AI - Intelligence Enhanced",
    description="🧠 Powered by patterns from thousands of cases",
    version="10.0-INTELLIGENCE"
)

# Load your original data
with open('data/simple_index.pkl', 'rb') as f:
    search_data = pickle.load(f)
    documents = search_data['documents']

# Load RAG engine
rag_engine = LegalRAG()

# LOAD THE CORPUS INTELLIGENCE!
try:
    with open('corpus_intelligence.json', 'r') as f:
        corpus_intelligence = json.load(f)
    print("✅ Loaded corpus intelligence!")
    print(f"   - {len(corpus_intelligence.get('winning_patterns', {}))} winning patterns")
    print(f"   - {len(corpus_intelligence.get('case_outcomes', []))} analyzed cases")
    print(f"   - Settlement data: ${corpus_intelligence.get('settlement_intelligence', {}).get('average', 0):,.0f} average")
except Exception as e:
    print(f"⚠️ Could not load corpus intelligence: {e}")
    corpus_intelligence = {}

class IntelligentPredictionRequest(BaseModel):
    case_details: str
    salary: Optional[float] = None
    years_service: Optional[int] = None

class IntelligentCasePredictor:
    def __init__(self, intelligence: Dict):
        self.intelligence = intelligence
        self.winning_patterns = intelligence.get('winning_patterns', {})
        self.settlement_data = intelligence.get('settlement_intelligence', {})
        self.judge_patterns = intelligence.get('judge_patterns', {})
        
    def predict_with_intelligence(self, case_details: str) -> Dict:
        """Use learned patterns for prediction"""
        case_lower = case_details.lower()
        
        # Base score
        score = 50
        intelligence_factors = []
        patterns_matched = []
        
        # Check against learned winning patterns
        for pattern, data in self.winning_patterns.items():
            pattern_keywords = {
                'no_warning': ['no warning', 'without warning', 'no prior warning'],
                'long_service': [r'\d+\s*years?', 'long service', 'loyal service'],
                'summary_dismissal': ['summary dismissal', 'immediate termination'],
                'serious_misconduct': ['serious misconduct', 'gross misconduct'],
                'procedural_fairness': ['no opportunity', 'unfair process', 'not given chance']
            }
            
            # Check if pattern present
            if pattern in pattern_keywords:
                for keyword in pattern_keywords[pattern]:
                    if re.search(keyword, case_lower):
                        # Use actual win rate from data
                        win_rate = data.get('win_rate', 0.5)
                        impact = data.get('impact', 0)
                        
                        score += impact * 100
                        
                        intelligence_factors.append({
                            'factor': pattern.replace('_', ' ').title(),
                            'impact': f"{impact*100:+.1f}%",
                            'confidence': f"Based on {data.get('occurrences', 0)} cases",
                            'historical_win_rate': f"{win_rate*100:.1f}%"
                        })
                        patterns_matched.append(pattern)
                        break
        
        # Limit score to reasonable range
        score = min(max(score, 5), 95)
        
        return {
            'intelligence_score': score,
            'patterns_matched': patterns_matched,
            'intelligence_factors': intelligence_factors,
            'confidence_level': self._calculate_confidence(patterns_matched),
            'based_on_cases': sum(
                self.winning_patterns.get(p, {}).get('occurrences', 0) 
                for p in patterns_matched
            )
        }
    
    def _calculate_confidence(self, patterns: List[str]) -> str:
        """Calculate confidence based on pattern matches"""
        total_cases = sum(
            self.winning_patterns.get(p, {}).get('occurrences', 0) 
            for p in patterns
        )
        
        if total_cases > 50 and len(patterns) > 2:
            return "VERY HIGH"
        elif total_cases > 20 or len(patterns) > 1:
            return "HIGH"
        elif total_cases > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def suggest_settlement(self, salary: float, years: int, case_strength: float) -> Dict:
        """Suggest settlement based on historical data"""
        
        if not self.settlement_data or not self.settlement_data.get('percentiles'):
            # Fallback calculation
            weekly = salary / 52
            return {
                'estimated_range': {
                    'low': weekly * 4,
                    'medium': weekly * 12,
                    'high': weekly * 26
                },
                'confidence': 'LOW - No historical data'
            }
        
        # Use percentiles from corpus
        percentiles = self.settlement_data['percentiles']
        
        # Map case strength to percentile
        if case_strength > 80:
            target = percentiles.get('75th', percentiles.get('median', 30000))
        elif case_strength > 60:
            target = percentiles.get('50th', percentiles.get('median', 20000))
        else:
            target = percentiles.get('25th', 10000)
        
        weekly = salary / 52
        
        return {
            'historical_settlement_data': {
                'average': self.settlement_data.get('average', 0),
                'median': self.settlement_data.get('median', 0),
                'based_on': f"{self.settlement_data.get('count', 0)} settlements"
            },
            'your_case_estimate': {
                'recommended_target': target,
                'weeks_equivalent': round(target / weekly, 1) if weekly > 0 else 0,
                'range': {
                    'minimum': percentiles.get('25th', 10000),
                    'likely': target,
                    'maximum': percentiles.get('75th', 50000)
                }
            },
            'strategy': self._settlement_strategy(case_strength, target, weekly)
        }
    
    def _settlement_strategy(self, strength: float, target: float, weekly: float) -> Dict:
        """Generate settlement strategy"""
        
        if strength > 75:
            return {
                'approach': 'Aggressive',
                'opening_demand': target * 1.5,
                'minimum_acceptable': target * 0.8,
                'rationale': 'Strong case - aim high'
            }
        else:
            return {
                'approach': 'Realistic',
                'opening_demand': target * 1.2,
                'minimum_acceptable': target * 0.6,
                'rationale': 'Moderate case - be flexible'
            }

# Initialize predictor
predictor = IntelligentCasePredictor(corpus_intelligence)

# ============= ENHANCED ENDPOINTS =============

@app.get("/")
async def root():
    return {
        "message": "🧠 Legal AI with Corpus Intelligence",
        "intelligence_loaded": bool(corpus_intelligence),
        "capabilities": {
            "patterns_learned": len(corpus_intelligence.get('winning_patterns', {})),
            "cases_analyzed": len(corpus_intelligence.get('case_outcomes', [])),
            "settlement_data": bool(corpus_intelligence.get('settlement_intelligence')),
            "judge_insights": len(corpus_intelligence.get('judge_patterns', {}))
        },
        "endpoints": {
            "/predict/intelligent": "Prediction using learned patterns",
            "/settlement/intelligent": "Settlement based on historical data",
            "/patterns": "View learned patterns",
            "/insights": "Corpus insights"
        }
    }

@app.post("/predict/intelligent")
async def intelligent_prediction(request: IntelligentPredictionRequest):
    """Predict using corpus intelligence"""
    
    # Get intelligence-based prediction
    intel_prediction = predictor.predict_with_intelligence(request.case_details)
    
    # Get settlement estimate if salary provided
    settlement = None
    if request.salary:
        settlement = predictor.suggest_settlement(
            request.salary,
            request.years_service or 2,
            intel_prediction['intelligence_score']
        )
    
    # Search for similar cases
    search_results = rag_engine.query(request.case_details, 3)
    
    return {
        'intelligence_prediction': intel_prediction,
        'settlement_recommendation': settlement,
        'similar_cases': search_results['sources'][:3],
        'explanation': self._generate_explanation(intel_prediction),
        'confidence_statement': self._confidence_statement(intel_prediction)
    }

def _generate_explanation(prediction: Dict) -> str:
    """Generate human-readable explanation"""
    
    score = prediction['intelligence_score']
    factors = prediction['intelligence_factors']
    
    if score > 75:
        strength = "strong"
    elif score > 50:
        strength = "moderate"
    else:
        strength = "weak"
    
    explanation = f"Based on analysis of {prediction['based_on_cases']} similar cases, this appears to be a {strength} case. "
    
    if factors:
        explanation += "Key factors: "
        explanation += ", ".join([f"{f['factor']} ({f['impact']})" for f in factors[:3]])
    
    return explanation

def _confidence_statement(prediction: Dict) -> str:
    """Generate confidence statement"""
    
    confidence = prediction['confidence_level']
    based_on = prediction['based_on_cases']
    
    if confidence == "VERY HIGH":
        return f"Very high confidence based on {based_on} similar cases with clear patterns."
    elif confidence == "HIGH":
        return f"High confidence based on {based_on} relevant cases."
    elif confidence == "MEDIUM":
        return f"Moderate confidence based on {based_on} cases."
    else:
        return "Limited historical data - recommendation should be verified."

@app.get("/patterns")
async def view_patterns():
    """View learned winning patterns"""
    
    patterns = corpus_intelligence.get('winning_patterns', {})
    
    # Sort by impact
    sorted_patterns = sorted(
        patterns.items(),
        key=lambda x: x[1].get('impact', 0),
        reverse=True
    )
    
    return {
        'winning_patterns': [
            {
                'pattern': pattern,
                'win_rate': f"{data.get('win_rate', 0)*100:.1f}%",
                'impact': f"{data.get('impact', 0)*100:+.1f}%",
                'based_on': f"{data.get('occurrences', 0)} cases",
                'classification': data.get('classification', 'unknown')
            }
            for pattern, data in sorted_patterns
        ],
        'total_patterns': len(patterns)
    }

@app.get("/insights")
async def corpus_insights():
    """Get insights from corpus analysis"""
    
    insights = corpus_intelligence.get('corpus_insights', {})
    
    return {
        'success_factors': insights.get('success_factors', []),
        'risk_factors': insights.get('risk_factors', []),
        'strategic_recommendations': insights.get('strategic_recommendations', []),
        'settlement_intelligence': corpus_intelligence.get('settlement_intelligence', {}),
        'judge_patterns': list(corpus_intelligence.get('judge_patterns', {}).keys())[:10]
    }

@app.post("/analyze/advanced")
async def advanced_analysis(request: IntelligentPredictionRequest):
    """Complete analysis with all intelligence"""
    
    # Intelligence prediction
    intel_pred = predictor.predict_with_intelligence(request.case_details)
    
    # Settlement recommendation
    settlement = None
    if request.salary:
        settlement = predictor.suggest_settlement(
            request.salary,
            request.years_service or 2,
            intel_pred['intelligence_score']
        )
    
    # Find precedents
    precedents = rag_engine.query(request.case_details, 5)
    
    # Check temporal patterns
    temporal = corpus_intelligence.get('temporal_patterns', {})
    current_year_trend = temporal.get('2023', {}).get('win_rate', 0.5) if temporal else 0.5
    
    return {
        'executive_summary': {
            'success_probability': f"{intel_pred['intelligence_score']}%",
            'confidence': intel_pred['confidence_level'],
            'recommended_action': 'Proceed with claim' if intel_pred['intelligence_score'] > 60 else 'Seek settlement',
            'potential_payout': settlement['your_case_estimate']['recommended_target'] if settlement else 'Unknown'
        },
        'detailed_analysis': {
            'intelligence_factors': intel_pred['intelligence_factors'],
            'patterns_matched': intel_pred['patterns_matched'],
            'based_on_cases': intel_pred['based_on_cases']
        },
        'settlement_guidance': settlement,
        'precedents': precedents['sources'][:3],
        'trends': {
            'current_year_win_rate': f"{current_year_trend*100:.1f}%",
            'historical_average': f"{corpus_intelligence.get('case_outcomes', [{}])[0].get('win_rate', 50)}%"
        }
    }

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 LEGAL AI WITH CORPUS INTELLIGENCE")
    print("=" * 60)
    
    if corpus_intelligence:
        print(f"✅ Loaded intelligence from {len(corpus_intelligence.get('case_outcomes', []))} cases")
        print(f"✅ {len(corpus_intelligence.get('winning_patterns', {}))} winning patterns identified")
        
        if corpus_intelligence.get('settlement_intelligence'):
            print(f"✅ Settlement data: ${corpus_intelligence['settlement_intelligence'].get('average', 0):,.0f} average")
    else:
        print("⚠️ Running without corpus intelligence")
    
    print("=" * 60)
    print("🌐 Starting on http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
