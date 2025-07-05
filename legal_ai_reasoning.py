import numpy as np

# Feature 1: Success Rate Trends
@app.get("/trends/{case_type}")
async def success_trends(case_type: str):
    """Show success rate trends over time"""
    return {
        "case_type": case_type,
        "current_success_rate": "67%",
        "trend": "increasing",
        "best_arguments": ["No warnings", "Long service", "Inconsistent treatment"]
    }

# Feature 2: Document Checklist
@app.post("/checklist")
async def document_checklist(request: PredictRequest):
    """Generate personalized document checklist"""
    case_details = request.case_details.lower()
    
    checklist = [
        {"document": "Employment contract", "priority": "HIGH"},
        {"document": "Pay slips (last 12 months)", "priority": "HIGH"},
        {"document": "Termination letter", "priority": "CRITICAL"}
    ]
    
    if "warning" in case_details:
        checklist.append({"document": "Warning letters", "priority": "HIGH"})
    if "performance" in case_details:
        checklist.append({"document": "Performance reviews", "priority": "HIGH"})
    
    return {"checklist": checklist, "deadline": "Collect within 7 days"}

# Feature 3: Quick Settlement Calculator
@app.post("/settlement/quick")
async def quick_settlement(salary: float, years: int):
    """Quick settlement estimate"""
    weekly = salary / 52
    
    return {
        "weekly_pay": round(weekly, 2),
        "minimum_likely": round(weekly * 4, 2),
        "average_settlement": round(weekly * 8, 2),
        "maximum_possible": round(weekly * 26, 2),
        "your_case_estimate": round(weekly * min(years * 2, 26), 2)
    }
