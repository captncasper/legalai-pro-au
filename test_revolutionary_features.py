#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Revolutionary Legal AI Features
Tests all endpoints with realistic legal scenarios
"""
import json
import requests
import time
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8001"

class LegalAITester:
    def __init__(self):
        self.test_results = {}
        self.recommendations = []
        
    def test_case_outcome_prediction(self):
        """Test case outcome prediction with various scenarios"""
        logger.info("🔮 Testing Case Outcome Prediction...")
        
        test_cases = [
            {
                "name": "Strong Negligence Case",
                "data": {
                    "case_type": "negligence",
                    "facts": "Plaintiff slipped on wet floor at Woolworths supermarket. No warning signs present despite store policy requiring them. CCTV shows employee mopping area 5 minutes before incident. Plaintiff suffered fractured hip requiring surgery and 6 months rehabilitation. Store manager admits warning signs were not placed due to staff shortage.",
                    "jurisdiction": "NSW",
                    "opposing_party_type": "corporation",
                    "claim_amount": 350000,
                    "evidence_strength": "strong"
                }
            },
            {
                "name": "Weak Contract Dispute",
                "data": {
                    "case_type": "contract",
                    "facts": "Verbal agreement to provide consulting services. No written contract. Client claims work was not completed to standard but cannot specify exact requirements. Consultant claims payment withheld without justification. No clear documentation of scope or deliverables.",
                    "jurisdiction": "VIC",
                    "opposing_party_type": "small_business",
                    "claim_amount": 25000,
                    "evidence_strength": "weak"
                }
            },
            {
                "name": "Employment Dismissal",
                "data": {
                    "case_type": "employment",
                    "facts": "Employee dismissed after 3 years employment. Given verbal warning 6 months prior for performance issues. No written warnings or performance improvement plan. Employee claims discrimination based on age (55 years old). Recent hire in same role is 28 years old with less experience.",
                    "jurisdiction": "QLD",
                    "opposing_party_type": "corporation",
                    "claim_amount": 75000,
                    "evidence_strength": "moderate"
                }
            },
            {
                "name": "Complex Constitutional Matter",
                "data": {
                    "case_type": "constitutional",
                    "facts": "State law conflicts with Commonwealth legislation regarding environmental protection. Mining company seeks to operate under state approval while Commonwealth agency claims jurisdiction under external affairs power. Constitutional validity of state law challenged.",
                    "jurisdiction": "WA",
                    "opposing_party_type": "government",
                    "claim_amount": 2000000,
                    "evidence_strength": "strong"
                }
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/v1/predict-outcome",
                    json=test_case["data"],
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "test_name": test_case["name"],
                        "success": True,
                        "response": result,
                        "analysis": self._analyze_outcome_prediction(result)
                    })
                    logger.info(f"✅ {test_case['name']}: Success")
                else:
                    results.append({
                        "test_name": test_case["name"],
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    })
                    logger.error(f"❌ {test_case['name']}: Failed")
                    
            except Exception as e:
                results.append({
                    "test_name": test_case["name"],
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"❌ {test_case['name']}: Exception - {e}")
        
        self.test_results["case_outcome_prediction"] = results
        return results
    
    def test_risk_analysis(self):
        """Test legal risk analysis with various document types"""
        logger.info("⚡ Testing Legal Risk Analysis...")
        
        test_documents = [
            {
                "name": "High Risk Construction Contract",
                "data": {
                    "document_text": """
                    CONSTRUCTION AGREEMENT
                    
                    The Contractor agrees to indemnify and hold harmless the Client from any and all claims, damages, losses, costs and expenses arising from or related to the performance of this agreement, regardless of whether such claims arise from the negligence of the Client or third parties.
                    
                    Contractor accepts unlimited liability for any breach of this contract including consequential damages, lost profits, and business interruption.
                    
                    This agreement may be terminated by Client at any time without cause upon 24 hours notice. Contractor shall have no right to compensation for work completed but not yet invoiced.
                    
                    All intellectual property created during the project, including derivative works and improvements, shall immediately become the sole property of Client.
                    
                    Contractor warrants that all work will be free from defects for a period of 10 years and agrees to remedy any defects at no cost to Client.
                    """,
                    "document_type": "contract",
                    "your_role": "contractor"
                }
            },
            {
                "name": "Employment Agreement Issues",
                "data": {
                    "document_text": """
                    EMPLOYMENT CONTRACT
                    
                    Employee acknowledges that they may be required to work up to 70 hours per week as business demands require.
                    
                    Employee agrees to a restraint of trade preventing them from working in the same industry for 3 years within 500km of any Company office after termination.
                    
                    Company may terminate employment immediately without notice or payment in lieu for any reason deemed appropriate by management.
                    
                    Employee waives all rights to unfair dismissal claims and agrees that any disputes must be resolved through binding arbitration.
                    
                    All overtime work is considered part of regular duties and no additional compensation will be provided.
                    """,
                    "document_type": "employment",
                    "your_role": "employee"
                }
            },
            {
                "name": "Moderate Risk Service Agreement",
                "data": {
                    "document_text": """
                    PROFESSIONAL SERVICES AGREEMENT
                    
                    Provider agrees to deliver consulting services with best efforts and in accordance with industry standards.
                    
                    Either party may terminate this agreement with 30 days written notice.
                    
                    Provider's liability shall be limited to the amount of fees paid under this agreement in the 12 months preceding any claim.
                    
                    Client retains ownership of all pre-existing intellectual property. New intellectual property developed collaboratively shall be jointly owned.
                    
                    Provider maintains professional indemnity insurance of $2 million.
                    """,
                    "document_type": "contract",
                    "your_role": "provider"
                }
            }
        ]
        
        results = []
        for test_doc in test_documents:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/v1/analyze-risk",
                    json=test_doc["data"],
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "test_name": test_doc["name"],
                        "success": True,
                        "response": result,
                        "analysis": self._analyze_risk_assessment(result)
                    })
                    logger.info(f"✅ {test_doc['name']}: Success")
                else:
                    results.append({
                        "test_name": test_doc["name"],
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    })
                    logger.error(f"❌ {test_doc['name']}: Failed")
                    
            except Exception as e:
                results.append({
                    "test_name": test_doc["name"],
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"❌ {test_doc['name']}: Exception - {e}")
        
        self.test_results["risk_analysis"] = results
        return results
    
    def test_settlement_analysis(self):
        """Test settlement analysis with various scenarios"""
        logger.info("💰 Testing Settlement Analysis...")
        
        test_settlements = [
            {
                "name": "High Value Personal Injury",
                "data": {
                    "case_type": "negligence",
                    "claim_amount": 1500000,
                    "liability_assessment": "clear_liability",
                    "jurisdiction": "NSW"
                }
            },
            {
                "name": "Commercial Dispute - Disputed Liability",
                "data": {
                    "case_type": "contract",
                    "claim_amount": 500000,
                    "liability_assessment": "disputed_liability",
                    "jurisdiction": "VIC"
                }
            },
            {
                "name": "Employment Matter with Contributory Factors",
                "data": {
                    "case_type": "employment",
                    "claim_amount": 120000,
                    "liability_assessment": "contributory_negligence",
                    "jurisdiction": "QLD"
                }
            },
            {
                "name": "Small Claims Court Matter",
                "data": {
                    "case_type": "contract",
                    "claim_amount": 15000,
                    "liability_assessment": "probable_liability",
                    "jurisdiction": "SA"
                }
            }
        ]
        
        results = []
        for test_settlement in test_settlements:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/v1/analyze-settlement",
                    json=test_settlement["data"],
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "test_name": test_settlement["name"],
                        "success": True,
                        "response": result,
                        "analysis": self._analyze_settlement_prediction(result)
                    })
                    logger.info(f"✅ {test_settlement['name']}: Success")
                else:
                    results.append({
                        "test_name": test_settlement["name"],
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    })
                    logger.error(f"❌ {test_settlement['name']}: Failed")
                    
            except Exception as e:
                results.append({
                    "test_name": test_settlement["name"],
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"❌ {test_settlement['name']}: Exception - {e}")
        
        self.test_results["settlement_analysis"] = results
        return results
    
    def test_api_health(self):
        """Test basic API health and info endpoints"""
        logger.info("🏥 Testing API Health...")
        
        try:
            response = requests.get(f"{BASE_URL}/api", timeout=5)
            if response.status_code == 200:
                api_info = response.json()
                return {
                    "success": True,
                    "api_info": api_info,
                    "corpus_size": api_info.get("corpus_size", 0),
                    "features_available": api_info.get("revolutionary_features", [])
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _analyze_outcome_prediction(self, result: Dict) -> Dict:
        """Analyze outcome prediction quality"""
        analysis = {"quality_score": 0, "issues": [], "strengths": []}
        
        if "case_outcome_analysis" in result:
            outcome = result["case_outcome_analysis"]
            
            # Check if success probability is reasonable
            prob = outcome.get("success_probability", 0)
            if 0 <= prob <= 100:
                analysis["strengths"].append("Reasonable probability range")
                analysis["quality_score"] += 20
            else:
                analysis["issues"].append("Probability outside valid range")
            
            # Check for legal analysis depth
            if "legal_analysis" in outcome:
                legal = outcome["legal_analysis"]
                if legal.get("legal_requirements"):
                    analysis["strengths"].append("Detailed legal requirements provided")
                    analysis["quality_score"] += 30
                if legal.get("elements_present") or legal.get("elements_weak"):
                    analysis["strengths"].append("Element-by-element analysis")
                    analysis["quality_score"] += 25
            
            # Check for comparable cases
            if outcome.get("similar_cases_found", 0) > 0:
                analysis["strengths"].append("Found comparable cases")
                analysis["quality_score"] += 25
            else:
                analysis["issues"].append("No similar cases found")
        
        return analysis
    
    def _analyze_risk_assessment(self, result: Dict) -> Dict:
        """Analyze risk assessment quality"""
        analysis = {"quality_score": 0, "issues": [], "strengths": []}
        
        if "risk_analysis" in result:
            risk = result["risk_analysis"]
            
            # Check risk identification
            risks = risk.get("identified_risks", [])
            if risks:
                analysis["strengths"].append(f"Identified {len(risks)} specific risks")
                analysis["quality_score"] += 30
                
                # Check for critical risks
                critical_risks = [r for r in risks if r.get("severity") == "CRITICAL"]
                if critical_risks:
                    analysis["strengths"].append("Correctly identified critical risks")
                    analysis["quality_score"] += 20
            else:
                analysis["issues"].append("No risks identified")
            
            # Check for amendments
            amendments = risk.get("recommended_amendments", [])
            if amendments:
                analysis["strengths"].append(f"Provided {len(amendments)} amendment recommendations")
                analysis["quality_score"] += 25
            
            # Check priority actions
            actions = risk.get("priority_actions", [])
            if actions:
                analysis["strengths"].append("Clear priority actions provided")
                analysis["quality_score"] += 25
        
        return analysis
    
    def _analyze_settlement_prediction(self, result: Dict) -> Dict:
        """Analyze settlement prediction quality"""
        analysis = {"quality_score": 0, "issues": [], "strengths": []}
        
        if "settlement_analysis" in result:
            settlement = result["settlement_analysis"]
            
            # Check settlement range
            if "settlement_range" in settlement.get("settlement_analysis", {}):
                analysis["strengths"].append("Comprehensive settlement range provided")
                analysis["quality_score"] += 30
            
            # Check negotiation strategy
            if "negotiation_strategy" in settlement:
                strategy = settlement["negotiation_strategy"]
                if strategy.get("opening_position") and strategy.get("minimum_acceptable"):
                    analysis["strengths"].append("Detailed negotiation strategy")
                    analysis["quality_score"] += 35
            
            # Check comparable cases
            comparables = settlement.get("comparable_cases", [])
            if comparables:
                analysis["strengths"].append(f"Found {len(comparables)} comparable cases")
                analysis["quality_score"] += 35
            else:
                analysis["issues"].append("No comparable cases found")
        
        return analysis
    
    def generate_recommendations(self):
        """Generate feature improvements and new feature recommendations"""
        logger.info("📋 Generating Recommendations...")
        
        recommendations = []
        
        # Analyze test results for patterns
        for feature, results in self.test_results.items():
            successful_tests = [r for r in results if r.get("success")]
            failed_tests = [r for r in results if not r.get("success")]
            
            if failed_tests:
                recommendations.append({
                    "type": "bug_fix",
                    "priority": "high",
                    "feature": feature,
                    "issue": f"{len(failed_tests)} tests failed",
                    "recommendation": f"Debug and fix {feature} endpoint reliability"
                })
            
            # Quality analysis
            if successful_tests:
                avg_quality = sum(
                    t.get("analysis", {}).get("quality_score", 0) 
                    for t in successful_tests
                ) / len(successful_tests)
                
                if avg_quality < 70:
                    recommendations.append({
                        "type": "enhancement",
                        "priority": "medium",
                        "feature": feature,
                        "issue": f"Average quality score: {avg_quality:.1f}",
                        "recommendation": f"Improve {feature} analysis depth and accuracy"
                    })
        
        # New feature recommendations
        new_features = [
            {
                "type": "new_feature",
                "priority": "high",
                "name": "Judge Analytics Engine",
                "description": "Analyze judge patterns, decision history, and preferences",
                "business_value": "Provides tactical advantage in court proceedings"
            },
            {
                "type": "new_feature",
                "priority": "high",
                "name": "Legal Strategy Generator",
                "description": "Generate complete legal strategies with timelines and resources",
                "business_value": "Automates strategy development, saves significant time"
            },
            {
                "type": "new_feature",
                "priority": "medium",
                "name": "Cost Prediction Engine",
                "description": "Predict legal costs by case type, complexity, and jurisdiction",
                "business_value": "Enables accurate budgeting and client cost estimates"
            },
            {
                "type": "new_feature",
                "priority": "medium",
                "name": "Document Automation",
                "description": "Auto-generate legal documents based on case analysis",
                "business_value": "Reduces document preparation time by 80%"
            },
            {
                "type": "new_feature", 
                "priority": "medium",
                "name": "Real-time Legal Updates",
                "description": "Monitor and analyze new case law and legislation",
                "business_value": "Ensures advice is based on latest legal developments"
            },
            {
                "type": "new_feature",
                "priority": "low",
                "name": "Client Communication Assistant",
                "description": "Generate client reports and explanations in plain English",
                "business_value": "Improves client experience and understanding"
            }
        ]
        
        recommendations.extend(new_features)
        
        # System improvements
        system_improvements = [
            {
                "type": "system_improvement",
                "priority": "high",
                "name": "Performance Optimization",
                "description": "Optimize corpus loading and search performance",
                "recommendation": "Implement caching and indexing improvements"
            },
            {
                "type": "system_improvement",
                "priority": "medium",
                "name": "Enhanced Error Handling",
                "description": "Better error messages and fallback mechanisms",
                "recommendation": "Add comprehensive error handling and user feedback"
            },
            {
                "type": "system_improvement",
                "priority": "medium",
                "name": "API Rate Limiting",
                "description": "Implement rate limiting for production use",
                "recommendation": "Add rate limiting and usage analytics"
            }
        ]
        
        recommendations.extend(system_improvements)
        self.recommendations = recommendations
        return recommendations
    
    def run_comprehensive_test(self):
        """Run all tests and generate complete report"""
        logger.info("🚀 Starting Comprehensive Legal AI Testing...")
        
        start_time = time.time()
        
        # Test API health first
        api_health = self.test_api_health()
        
        if not api_health.get("success"):
            logger.error("❌ API Health Check Failed - Aborting tests")
            return {"error": "API not available", "health_check": api_health}
        
        logger.info("✅ API Health Check Passed")
        
        # Run all feature tests
        outcome_results = self.test_case_outcome_prediction()
        risk_results = self.test_risk_analysis()
        settlement_results = self.test_settlement_analysis()
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        end_time = time.time()
        
        # Compile comprehensive report
        report = {
            "test_summary": {
                "total_duration": f"{end_time - start_time:.2f} seconds",
                "api_health": api_health,
                "total_tests": sum(len(results) for results in self.test_results.values()),
                "successful_tests": sum(
                    len([r for r in results if r.get("success")])
                    for results in self.test_results.values()
                ),
                "failed_tests": sum(
                    len([r for r in results if not r.get("success")])
                    for results in self.test_results.values()
                )
            },
            "feature_test_results": self.test_results,
            "recommendations": recommendations,
            "next_steps": self._generate_next_steps()
        }
        
        return report
    
    def _generate_next_steps(self):
        """Generate prioritized next steps"""
        return [
            "1. Address any failed tests and critical bugs",
            "2. Implement Judge Analytics Engine (high business value)",
            "3. Add Legal Strategy Generator functionality", 
            "4. Optimize performance for larger corpus sizes",
            "5. Enhance error handling and user experience",
            "6. Add comprehensive API documentation",
            "7. Implement cost prediction capabilities",
            "8. Add real-time legal update monitoring"
        ]

def main():
    """Run the comprehensive test suite"""
    tester = LegalAITester()
    
    # Give server time to start
    print("Waiting for server to be ready...")
    time.sleep(3)
    
    # Run comprehensive tests
    report = tester.run_comprehensive_test()
    
    # Save detailed report
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("🔍 REVOLUTIONARY LEGAL AI TEST REPORT")
    print("="*80)
    
    if "error" in report:
        print(f"❌ Testing failed: {report['error']}")
        return
    
    summary = report["test_summary"]
    print(f"⏱️  Total Duration: {summary['total_duration']}")
    print(f"✅ Successful Tests: {summary['successful_tests']}")
    print(f"❌ Failed Tests: {summary['failed_tests']}")
    print(f"📊 Success Rate: {summary['successful_tests']/summary['total_tests']*100:.1f}%")
    
    print(f"\n📋 Generated {len(report['recommendations'])} recommendations")
    
    # Show top recommendations
    high_priority = [r for r in report['recommendations'] if r.get('priority') == 'high']
    if high_priority:
        print("\n🚨 HIGH PRIORITY RECOMMENDATIONS:")
        for rec in high_priority[:5]:
            print(f"   • {rec.get('name', rec.get('recommendation', 'Unknown'))}")
    
    print(f"\n📄 Detailed report saved to: test_report.json")
    print("="*80)

if __name__ == "__main__":
    main()