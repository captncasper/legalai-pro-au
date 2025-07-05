#!/usr/bin/env python3
"""
ULTIMATE LEGAL AI - ULTRA SMART EDITION
Maximum intelligence with advanced AI features
"""

from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Any
import pickle
import re
from collections import Counter, defaultdict
import uvicorn
from legal_rag import LegalRAG
from datetime import datetime, timedelta
import json
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import hashlib
import io

app = FastAPI(
    title="Ultimate Legal AI - ULTRA SMART",
    description="🧠 Maximum intelligence: Pattern recognition, Auto-drafting, Risk analysis, Strategic planning",
    version="8.0-ULTRA"
)

# Load data
with open('data/simple_index.pkl', 'rb') as f:
    search_data = pickle.load(f)
    documents = search_data['documents']

# Pre-built indexes
citation_index = {doc.get('metadata', {}).get('citation', ''): i 
                  for i, doc in enumerate(documents) if doc.get('metadata', {}).get('citation')}
type_index = defaultdict(list)
for i, doc in enumerate(documents):
    doc_type = doc.get('metadata', {}).get('type', 'unknown')
    type_index[doc_type].append(i)

# Pattern recognition database
case_patterns = defaultdict(list)
for i, doc in enumerate(documents):
    text_lower = doc['text'].lower()
    if 'unfair' in text_lower and 'dismissal' in text_lower:
        case_patterns['unfair_dismissal'].append(i)
    if 'discriminat' in text_lower:
        case_patterns['discrimination'].append(i)
    if 'breach' in text_lower and 'contract' in text_lower:
        case_patterns['breach_contract'].append(i)

# Initialize engines
executor = ThreadPoolExecutor(max_workers=6)
rag_engine = LegalRAG()

# ============= PATTERN RECOGNITION ENGINE =============
class PatternRecognitionEngine:
    def __init__(self):
        self.patterns = {
            'winning_patterns': {
                'no_warning_termination': r'no\s*(?:prior\s*)?warning.{0,50}terminat',
                'long_service_dismissal': r'(?:\d+\s*years?|long\s*service).{0,50}dismiss',
                'discrimination_evidence': r'(?:treat\w*\s*different|single\w*\s*out|target\w*)',
                'procedural_unfairness': r'(?:no\s*opportunity|not\s*given\s*chance|unfair\s*process)',
                'constructive_dismissal': r'(?:forced\s*to\s*resign|no\s*choice|intolerable)',
            },
            'losing_patterns': {
                'serious_misconduct': r'(?:theft|fraud|violence|serious\s*misconduct)',
                'poor_performance': r'(?:performance\s*manage|warnings?\s*given|improvement\s*plan)',
                'genuine_redundancy': r'(?:genuine\s*redundancy|business\s*restructur|economic)',
            },
            'evidence_quality': {
                'strong': r'(?:email|letter|document|written|recorded)',
                'moderate': r'(?:witness|saw|heard|told)',
                'weak': r'(?:believe|think|feel|seemed)',
            }
        }
    
    def analyze_patterns(self, text: str) -> Dict:
        text_lower = text.lower()
        
        # Find all patterns
        found_patterns = {
            'winning': [],
            'losing': [],
            'evidence': []
        }
        
        # Check winning patterns
        for pattern_name, pattern in self.patterns['winning_patterns'].items():
            if re.search(pattern, text_lower):
                found_patterns['winning'].append({
                    'pattern': pattern_name,
                    'strength': 'HIGH',
                    'impact': '+20-30%'
                })
        
        # Check losing patterns
        for pattern_name, pattern in self.patterns['losing_patterns'].items():
            if re.search(pattern, text_lower):
                found_patterns['losing'].append({
                    'pattern': pattern_name,
                    'strength': 'HIGH',
                    'impact': '-20-40%'
                })
        
        # Assess evidence quality
        evidence_score = 0
        for quality, pattern in self.patterns['evidence_quality'].items():
            matches = len(re.findall(pattern, text_lower))
            if quality == 'strong':
                evidence_score += matches * 3
            elif quality == 'moderate':
                evidence_score += matches * 2
            else:
                evidence_score += matches
        
        # Find similar cases
        similar_cases = self._find_similar_patterns(text_lower)
        
        return {
            'patterns_found': found_patterns,
            'evidence_score': min(100, evidence_score * 10),
            'pattern_match_score': len(found_patterns['winning']) * 25 - len(found_patterns['losing']) * 30,
            'similar_cases': similar_cases[:3],
            'strategic_insights': self._generate_insights(found_patterns, evidence_score)
        }
    
    def _find_similar_patterns(self, text: str) -> List[Dict]:
        """Find cases with similar patterns"""
        text_words = set(text.split())
        similarities = []
        
        # Sample first 100 docs for speed
        for i in range(min(100, len(documents))):
            doc = documents[i]
            doc_words = set(doc['text'].lower().split())
            
            # Jaccard similarity
            similarity = len(text_words & doc_words) / len(text_words | doc_words)
            
            if similarity > 0.3:  # Threshold
                similarities.append({
                    'citation': doc.get('metadata', {}).get('citation', 'Unknown'),
                    'similarity': round(similarity * 100, 1),
                    'relevance': 'HIGH' if similarity > 0.5 else 'MEDIUM'
                })
        
        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    
    def _generate_insights(self, patterns: Dict, evidence_score: int) -> List[str]:
        insights = []
        
        if len(patterns['winning']) > len(patterns['losing']):
            insights.append("✅ Strong pattern match - similar cases often succeed")
        
        if evidence_score > 70:
            insights.append("📊 Excellent evidence quality - well documented")
        elif evidence_score > 40:
            insights.append("📝 Moderate evidence - consider gathering more documentation")
        else:
            insights.append("⚠️ Weak evidence - urgent need for supporting documents")
        
        if patterns['winning'] and not patterns['losing']:
            insights.append("🎯 Ideal case profile - high success probability")
        
        return insights

# ============= AUTO DOCUMENT GENERATOR =============
class AutoDocumentGenerator:
    def __init__(self):
        self.templates = {
            'f8c': self._f8c_template,
            'witness_statement': self._witness_template,
            'settlement_letter': self._settlement_template,
            'timeline': self._timeline_template,
            'evidence_list': self._evidence_template
        }
    
    def generate_suite(self, case_details: str, case_analysis: Dict) -> Dict:
        """Generate complete document suite based on case"""
        
        # Parse key information
        info = self._extract_info(case_details)
        
        # Generate all relevant documents
        documents = {}
        
        # Always generate F8C for unfair dismissal
        if 'unfair_dismissal' in case_analysis.get('claims', []):
            documents['f8c'] = self.templates['f8c'](info, case_analysis)
        
        # Generate timeline
        documents['timeline'] = self.templates['timeline'](info, case_details)
        
        # Generate evidence checklist
        documents['evidence_list'] = self.templates['evidence_list'](case_analysis)
        
        # Generate settlement letter if strong case
        if case_analysis.get('success_probability', 0) > 60:
            documents['settlement_letter'] = self.templates['settlement_letter'](info, case_analysis)
        
        return documents
    
    def _extract_info(self, text: str) -> Dict:
        """Extract key information from case details"""
        info = {
            'dismissal_date': 'Recently' if 'recent' in text.lower() else '[Date]',
            'years_service': None,
            'salary': None,
            'employer': '[Employer Name]',
            'position': '[Position]'
        }
        
        # Extract years
        years_match = re.search(r'(\d+)\s*years?', text)
        if years_match:
            info['years_service'] = years_match.group(1)
        
        # Extract salary
        salary_match = re.search(r'\$?([\d,]+)(?:k|K|\s*(?:thousand|per year|annually))?', text)
        if salary_match:
            salary_str = salary_match.group(1).replace(',', '')
            info['salary'] = int(salary_str) * (1000 if 'k' in text.lower() else 1)
        
        return info
    
    def _f8c_template(self, info: Dict, analysis: Dict) -> str:
        return f"""FORM F8C - UNFAIR DISMISSAL APPLICATION
[AUTO-GENERATED DRAFT - Review before submission]

1. APPLICANT DETAILS
Name: [Your Full Name]
Email: [Your Email]
Phone: [Your Phone]

2. EMPLOYER DETAILS  
Organisation: {info['employer']}
Position held: {info['position']}

3. EMPLOYMENT DETAILS
Years of service: {info.get('years_service', '[Years]')}
Dismissal date: {info['dismissal_date']}

4. GROUNDS FOR APPLICATION
Based on case analysis (Success probability: {analysis.get('success_probability', 'Unknown')}%):

The dismissal was:
☑ Harsh - {', '.join(f['pattern'] for f in analysis.get('patterns_found', {}).get('winning', [])[:2])}
☑ Unjust - No valid reason provided
☑ Unreasonable - Disproportionate to any alleged conduct

5. KEY FACTS
{self._generate_key_facts(analysis)}

6. REMEDY SOUGHT
☐ Reinstatement
☑ Compensation (recommended based on analysis)

URGENT: File within 21 days of dismissal
"""
    
    def _witness_template(self, info: Dict, analysis: Dict) -> str:
        return """WITNESS STATEMENT TEMPLATE
[Customize with specific details]

I, [Witness Name], of [Address], state:

1. I have worked with [Applicant] for [period] as [relationship].

2. KEY OBSERVATIONS:
   - [Specific incident/behavior witnessed]
   - [Dates and times if known]
   - [Other relevant observations]

3. In my opinion, [supporting statement].

Signed: _____________ Date: _______
"""
    
    def _settlement_template(self, info: Dict, analysis: Dict) -> str:
        return f"""WITHOUT PREJUDICE - SETTLEMENT PROPOSAL

Dear [Employer],

Re: [Your Name] - Settlement Proposal

Success Analysis: {analysis.get('success_probability', 70)}% likelihood of success at FWC

We propose the following settlement:
- Payment of [calculated amount based on analysis]
- Agreed reference
- No admission of liability

This represents a fair commercial resolution avoiding:
- Legal costs (estimated $20,000-50,000)
- Management time
- Reputational risk

Valid for 7 days.

Yours sincerely,
[Your Name]
"""
    
    def _timeline_template(self, info: Dict, text: str) -> str:
        events = []
        
        # Extract events from text
        if 'start' in text.lower():
            events.append("Employment commenced")
        if 'warning' in text.lower():
            events.append("Warning issued (if any)")
        if 'dismiss' in text.lower() or 'terminat' in text.lower():
            events.append("Dismissal/Termination")
        
        return f"""CHRONOLOGICAL TIMELINE

{chr(10).join(f'[Date] - {event}' for event in events)}

Key patterns identified:
- {info.get('years_service', 'Multiple')} years of service
- Dismissal circumstances: [Detail the final incident]
- Notice given: [Yes/No]
- Final pay: [Received/Outstanding]

CRITICAL DATES:
- Dismissal: {info['dismissal_date']}
- FWC deadline: [21 days from dismissal]
- Evidence collection deadline: [ASAP]
"""
    
    def _evidence_template(self, analysis: Dict) -> str:
        return """EVIDENCE CHECKLIST
[✓] = Have  [✗] = Need  [?] = Check

CRITICAL DOCUMENTS:
[ ] Employment contract
[ ] Termination letter/email
[ ] Pay slips (last 12 months)
[ ] Job description

IMPORTANT EVIDENCE:
[ ] Performance reviews
[ ] Emails about performance
[ ] Warning letters (if any)
[ ] Company policies
[ ] Comparator evidence

HELPFUL EVIDENCE:
[ ] Witness contact list
[ ] Medical certificates
[ ] Awards/commendations
[ ] Training records

DIGITAL EVIDENCE:
[ ] Email backups
[ ] Text messages
[ ] Teams/Slack messages
[ ] Calendar entries

Action: Organize chronologically in folders
"""
    
    def _generate_key_facts(self, analysis: Dict) -> str:
        facts = []
        
        patterns = analysis.get('patterns_found', {})
        if patterns.get('winning'):
            facts.extend([p['pattern'].replace('_', ' ').title() for p in patterns['winning'][:3]])
        
        return '\n'.join(f'• {fact}' for fact in facts) if facts else '• [List key facts supporting your case]'

# ============= RISK ANALYSIS ENGINE =============
class RiskAnalysisEngine:
    def analyze_risks(self, case_analysis: Dict, employer_type: str = 'unknown') -> Dict:
        risks = {
            'legal_risks': [],
            'financial_risks': [],
            'career_risks': [],
            'time_risks': []
        }
        
        success_prob = case_analysis.get('success_probability', 50)
        
        # Legal risks
        if success_prob < 40:
            risks['legal_risks'].append({
                'risk': 'Adverse costs order',
                'probability': 'MEDIUM',
                'impact': 'HIGH',
                'mitigation': 'Consider settlement or discontinuance'
            })
        
        if 'small business' in employer_type.lower():
            risks['legal_risks'].append({
                'risk': 'Small business exemption',
                'probability': 'HIGH',
                'impact': 'CRITICAL',
                'mitigation': 'Check employee count carefully'
            })
        
        # Financial risks
        risks['financial_risks'].append({
            'risk': 'Legal fees',
            'range': '$0-5000 (self-represented) to $20,000+ (lawyer)',
            'mitigation': 'Consider no-win-no-fee arrangements'
        })
        
        # Career risks
        if employer_type in ['government', 'large_corporate']:
            risks['career_risks'].append({
                'risk': 'Industry reputation',
                'probability': 'LOW-MEDIUM',
                'mitigation': 'Confidential settlement clause'
            })
        
        # Time investment
        risks['time_risks'] = {
            'conciliation': '2-3 hours',
            'hearing_prep': '20-40 hours',
            'hearing': '1-3 days',
            'total_duration': '3-6 months typical'
        }
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(risks, success_prob)
        
        return {
            'risk_assessment': risks,
            'overall_risk_level': risk_score['level'],
            'risk_score': risk_score['score'],
            'recommendation': risk_score['recommendation'],
            'mitigation_strategies': self._generate_mitigation_strategies(risks, success_prob)
        }
    
    def _calculate_risk_score(self, risks: Dict, success_prob: int) -> Dict:
        # Simple scoring algorithm
        score = 100 - success_prob
        
        # Adjust for risks
        score += len(risks['legal_risks']) * 10
        score += len(risks['career_risks']) * 5
        
        if score < 30:
            return {
                'score': score,
                'level': 'LOW',
                'recommendation': 'Proceed with confidence'
            }
        elif score < 60:
            return {
                'score': score,
                'level': 'MEDIUM',
                'recommendation': 'Proceed with caution - consider settlement'
            }
        else:
            return {
                'score': score,
                'level': 'HIGH',
                'recommendation': 'High risk - strongly consider alternatives'
            }
    
    def _generate_mitigation_strategies(self, risks: Dict, success_prob: int) -> List[str]:
        strategies = []
        
        if success_prob > 70:
            strategies.append("💪 Strong case - be confident in negotiations")
        
        if risks['legal_risks']:
            strategies.append("⚖️ Consider fixed-fee legal advice for risk mitigation")
        
        strategies.extend([
            "📝 Document everything meticulously",
            "🤝 Keep settlement door open",
            "⏰ Act quickly to preserve evidence",
            "👥 Secure witness statements early"
        ])
        
        return strategies

# ============= STRATEGIC PLANNER =============
class StrategicPlanner:
    def create_battle_plan(self, case_analysis: Dict, risk_analysis: Dict, timeline: Dict) -> Dict:
        """Create comprehensive legal strategy"""
        
        success_prob = case_analysis.get('success_probability', 50)
        risk_level = risk_analysis.get('overall_risk_level', 'MEDIUM')
        
        # Determine strategy
        if success_prob > 75 and risk_level == 'LOW':
            strategy = 'aggressive'
        elif success_prob > 50:
            strategy = 'balanced'
        else:
            strategy = 'defensive'
        
        plan = {
            'strategy': strategy,
            'phases': self._create_phases(strategy, timeline),
            'negotiation_approach': self._negotiation_strategy(success_prob, strategy),
            'communication_plan': self._communication_plan(strategy),
            'contingencies': self._contingency_plans(strategy, risk_level),
            'success_metrics': self._define_success_metrics(strategy)
        }
        
        return plan
    
    def _create_phases(self, strategy: str, timeline: Dict) -> List[Dict]:
        phases = []
        
        # Phase 1: Immediate (0-7 days)
        phases.append({
            'phase': 'Immediate Action',
            'duration': '0-7 days',
            'actions': [
                '📋 File F8C application' if strategy != 'defensive' else '📞 Seek legal advice',
                '📄 Gather all documents',
                '👥 Contact key witnesses',
                '💾 Backup all digital evidence'
            ],
            'critical': True
        })
        
        # Phase 2: Preparation (1-4 weeks)
        phases.append({
            'phase': 'Preparation',
            'duration': '1-4 weeks',
            'actions': [
                '📊 Prepare evidence bundle',
                '✍️ Draft witness statements',
                '💰 Calculate losses precisely',
                '🎯 Refine legal arguments'
            ]
        })
        
        # Phase 3: Conciliation (4-8 weeks)
        phases.append({
            'phase': 'Conciliation',
            'duration': '4-8 weeks',
            'actions': [
                '🤝 Prepare settlement positions',
                '📈 Develop BATNA',
                '🎭 Practice conciliation approach',
                '📋 Prepare conciliation brief'
            ]
        })
        
        return phases
    
    def _negotiation_strategy(self, success_prob: int, strategy: str) -> Dict:
        if strategy == 'aggressive':
            return {
                'opening_position': 'Maximum compensation + reinstatement',
                'target': '20-26 weeks pay',
                'minimum': '12 weeks pay',
                'tactics': ['Highlight strength', 'Press precedents', 'Time pressure'],
                'style': 'Confident and firm'
            }
        elif strategy == 'balanced':
            return {
                'opening_position': '26 weeks compensation',
                'target': '12-16 weeks pay',
                'minimum': '8 weeks pay',
                'tactics': ['Build rapport', 'Find middle ground', 'Package deal'],
                'style': 'Reasonable but firm'
            }
        else:
            return {
                'opening_position': 'Open to discussion',
                'target': '8-12 weeks pay',
                'minimum': '4 weeks pay',
                'tactics': ['Minimize conflict', 'Quick resolution', 'Face-saving'],
                'style': 'Conciliatory'
            }
    
    def _communication_plan(self, strategy: str) -> Dict:
        return {
            'internal': {
                'family': 'Keep informed of major developments',
                'witnesses': 'Regular updates, maintain enthusiasm',
                'support': 'Engage counselor if needed'
            },
            'external': {
                'opponent': 'All communication in writing' if strategy == 'aggressive' else 'Professional, door open',
                'commission': 'Prompt, professional responses',
                'media': 'No comment unless strategic advantage'
            }
        }
    
    def _contingency_plans(self, strategy: str, risk_level: str) -> List[Dict]:
        plans = []
        
        plans.append({
            'scenario': 'Settlement rejected',
            'response': 'Proceed to hearing' if strategy == 'aggressive' else 'Improve offer'
        })
        
        plans.append({
            'scenario': 'New evidence emerges',
            'response': 'Immediately assess impact and adjust'
        })
        
        if risk_level == 'HIGH':
            plans.append({
                'scenario': 'Case weakens',
                'response': 'Quick settlement on best terms available'
            })
        
        return plans
    
    def _define_success_metrics(self, strategy: str) -> Dict:
        return {
            'primary': 'Financial compensation' if strategy != 'aggressive' else 'Reinstatement or max compensation',
            'secondary': ['Clean reference', 'Quick resolution', 'Costs minimized'],
            'acceptable_outcomes': {
                'best': 'Full demands met',
                'good': 'Target compensation achieved',
                'acceptable': 'Above minimum threshold',
                'walk_away': 'Below minimum or admission required'
            }
        }

# ============= REAL-TIME MONITORING =============
class CaseMonitor:
    def __init__(self):
        self.active_cases = {}
    
    async def track_case(self, case_id: str, case_details: Dict):
        """Track case progress in real-time"""
        self.active_cases[case_id] = {
            'started': datetime.now(),
            'status': 'active',
            'milestones': [],
            'alerts': []
        }
        
        # Set up deadline monitoring
        asyncio.create_task(self._monitor_deadlines(case_id, case_details))
    
    async def _monitor_deadlines(self, case_id: str, case_details: Dict):
        """Background task to monitor deadlines"""
        while self.active_cases.get(case_id, {}).get('status') == 'active':
            # Check deadlines
            alerts = []
            
            dismissal_date = case_details.get('dismissal_date')
            if dismissal_date:
                days_left = 21 - (datetime.now() - dismissal_date).days
                if days_left <= 3 and days_left > 0:
                    alerts.append({
                        'type': 'CRITICAL',
                        'message': f'Only {days_left} days left to file!',
                        'action': 'File F8C immediately'
                    })
            
            if alerts:
                self.active_cases[case_id]['alerts'] = alerts
            
            await asyncio.sleep(3600)  # Check hourly

# Initialize all engines
pattern_engine = PatternRecognitionEngine()
doc_generator = AutoDocumentGenerator()
risk_engine = RiskAnalysisEngine()
strategic_planner = StrategicPlanner()
case_monitor = CaseMonitor()

# ============= ULTRA SMART ENDPOINTS =============

@app.get("/")
async def root():
    return {
        "message": "�� Ultimate Legal AI - ULTRA SMART Edition",
        "version": "8.0-ULTRA",
        "intelligence_features": {
            "pattern_recognition": "Identifies winning/losing patterns",
            "auto_documents": "Generates complete document suites",
            "risk_analysis": "Comprehensive risk assessment",
            "strategic_planning": "Battle-tested legal strategies",
            "real_time_monitoring": "Deadline and progress tracking"
        },
        "endpoints": {
            "/analyze/ultra": "🧠 Ultra-smart analysis with all features",
            "/patterns/analyze": "🔍 Pattern recognition analysis",
            "/documents/generate-suite": "📄 Auto-generate document suite",
            "/risk/assess": "⚠️ Comprehensive risk assessment",
            "/strategy/create": "🎯 Strategic battle plan",
            "/monitor/start": "📊 Real-time case monitoring"
        }
    }

@app.post("/analyze/ultra")
async def ultra_analysis(
    case_details: str,
    salary: Optional[float] = None,
    employer_type: str = "unknown",
    generate_documents: bool = True
):
    """Ultra-smart analysis with all AI features"""
    
    # Run everything in parallel
    tasks = []
    
    # All previous analyses
    tasks.append(parallel_analysis(case_details, salary))
    
    # New AI features
    tasks.append(asyncio.to_thread(pattern_engine.analyze_patterns, case_details))
    
    results = await asyncio.gather(*tasks)
    
    base_analysis = results[0]
    pattern_analysis = results[1]
    
    # Combine analyses for super intelligence
    combined_analysis = {
        **base_analysis,
        'pattern_analysis': pattern_analysis,
        'combined_success_score': (
            base_analysis['reasoning']['success_probability'] + 
            min(100, 50 + pattern_analysis['pattern_match_score'])
        ) / 2
    }
    
    # Risk assessment
    risk_analysis = risk_engine.analyze_risks(combined_analysis['reasoning'], employer_type)
    
    # Strategic planning
    timeline = {'dismissal_date': datetime.now()}  # Simplified
    strategy = strategic_planner.create_battle_plan(
        combined_analysis['reasoning'],
        risk_analysis,
        timeline
    )
    
    # Document generation if requested
    documents = {}
    if generate_documents:
        documents = doc_generator.generate_suite(case_details, combined_analysis['reasoning'])
    
    # Create case monitoring
    case_id = hashlib.md5(case_details.encode()).hexdigest()[:8]
    await case_monitor.track_case(case_id, {'dismissal_date': datetime.now()})
    
    return {
        'case_id': case_id,
        'ultra_analysis': combined_analysis,
        'risk_assessment': risk_analysis,
        'strategic_plan': strategy,
        'documents_generated': list(documents.keys()) if documents else [],
        'monitoring': {
            'status': 'active',
            'case_id': case_id,
            'next_check': 'hourly'
        },
        'executive_summary': {
            'success_probability': f"{combined_analysis['combined_success_score']:.1f}%",
            'risk_level': risk_analysis['overall_risk_level'],
            'strategy': strategy['strategy'],
            'next_action': strategy['phases'][0]['actions'][0] if strategy['phases'] else 'Review analysis'
        }
    }

@app.post("/patterns/analyze")
async def analyze_patterns(text: str):
    """Deep pattern analysis"""
    return pattern_engine.analyze_patterns(text)

@app.post("/documents/generate-suite")
async def generate_documents(case_details: str, case_analysis: Optional[Dict] = None):
    """Generate complete document suite"""
    if not case_analysis:
        # Run quick analysis
        case_analysis = LegalReasoningEngineOptimized().analyze(case_details)
    
    documents = doc_generator.generate_suite(case_details, case_analysis)
    
    return {
        'documents_generated': list(documents.keys()),
        'documents': documents,
        'download_ready': True,
        'next_steps': [
            'Review all documents carefully',
            'Customize with your specific details',
            'File F8C within deadline'
        ]
    }

@app.post("/risk/assess")
async def assess_risk(
    case_analysis: Dict,
    employer_type: str = "unknown",
    employer_size: Optional[int] = None
):
    """Comprehensive risk assessment"""
    if employer_size and employer_size < 15:
        employer_type = "small_business"
    
    return risk_engine.analyze_risks(case_analysis, employer_type)

@app.post("/strategy/create")
async def create_strategy(
    case_analysis: Dict,
    risk_analysis: Dict,
    timeline: Dict,
    preferences: Optional[Dict] = None
):
    """Create strategic battle plan"""
    strategy = strategic_planner.create_battle_plan(case_analysis, risk_analysis, timeline)
    
    if preferences:
        # Adjust strategy based on user preferences
        if preferences.get('avoid_hearing'):
            strategy['negotiation_approach']['style'] = 'Settlement focused'
    
    return strategy

@app.get("/monitor/{case_id}")
async def get_monitoring(case_id: str):
    """Get real-time monitoring status"""
    if case_id not in case_monitor.active_cases:
        raise HTTPException(404, "Case not found")
    
    return {
        'case_id': case_id,
        'monitoring': case_monitor.active_cases[case_id],
        'current_alerts': case_monitor.active_cases[case_id].get('alerts', [])
    }

# File upload endpoint for document analysis
@app.post("/analyze/document")
async def analyze_document(file: UploadFile = File(...)):
    """Analyze uploaded legal document"""
    contents = await file.read()
    text = contents.decode('utf-8', errors='ignore')
    
    # Quick pattern analysis
    patterns = pattern_engine.analyze_patterns(text)
    
    return {
        'filename': file.filename,
        'document_type': 'termination_letter' if 'terminat' in text.lower() else 'unknown',
        'patterns_found': patterns['patterns_found'],
        'evidence_quality': patterns['evidence_score'],
        'recommendations': patterns['strategic_insights']
    }

# WebSocket for real-time updates
@app.websocket("/ws/{case_id}")
async def websocket_monitor(websocket: WebSocket, case_id: str):
    await websocket.accept()
    
    try:
        while True:
            # Send updates
            if case_id in case_monitor.active_cases:
                await websocket.send_json({
                    'case_id': case_id,
                    'alerts': case_monitor.active_cases[case_id].get('alerts', []),
                    'status': case_monitor.active_cases[case_id].get('status')
                })
            
            await asyncio.sleep(30)  # Update every 30 seconds
    except:
        pass
    finally:
        await websocket.close()

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 ULTIMATE LEGAL AI - ULTRA SMART v8.0")
    print("=" * 60)
    print("✅ Pattern Recognition Engine")
    print("✅ Auto Document Generation")
    print("✅ Risk Analysis System")
    print("✅ Strategic Planning Module")
    print("✅ Real-time Monitoring")
    print("✅ File Upload Analysis")
    print("✅ WebSocket Updates")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
