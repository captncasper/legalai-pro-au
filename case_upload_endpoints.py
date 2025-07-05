#!/usr/bin/env python3
"""
Case upload endpoints to add to your unified system
"""

from fastapi import UploadFile, File, Form, HTTPException
from typing import Optional
import json
from datetime import datetime
import re

# Add these imports to your unified system
# from fastapi import UploadFile, File, Form

# Add these endpoints to your unified_with_scraping.py:

print('''
# ===== CASE UPLOAD ENDPOINTS =====

@app.post("/api/v1/cases/upload")
async def upload_case(
    citation: str = Form(...),
    case_name: str = Form(...),
    text: str = Form(...),
    outcome: str = Form("unknown"),
    jurisdiction: str = Form("nsw"),
    year: Optional[int] = Form(None),
    court: Optional[str] = Form(None),
    judge: Optional[str] = Form(None),
    catchwords: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Upload a new case to the corpus"""
    try:
        # If file is provided, extract text from it
        if file:
            content = await file.read()
            if file.filename.endswith('.txt'):
                text = content.decode('utf-8')
            elif file.filename.endswith('.json'):
                data = json.loads(content)
                text = data.get('text', data.get('judgment', ''))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type. Use .txt or .json")
        
        # Extract year from citation if not provided
        if not year:
            year_match = re.search(r'\[(\d{4})\]', citation)
            if year_match:
                year = int(year_match.group(1))
            else:
                year = datetime.now().year
        
        # Extract court from citation if not provided
        if not court:
            court_match = re.search(r'\[(\d{4})\]\s*(\w+)', citation)
            if court_match:
                court = court_match.group(2)
        
        # Create new case object
        new_case = {
            'citation': citation,
            'case_name': case_name,
            'text': text[:10000],  # Limit text length
            'outcome': outcome,
            'jurisdiction': jurisdiction,
            'year': year,
            'court': court or 'Unknown',
            'judge': judge,
            'catchwords': catchwords,
            'source': 'manual_upload',
            'uploaded_at': datetime.now().isoformat(),
            'factors': []  # Will be populated by analysis
        }
        
        # Add to corpus
        unified_ai.corpus.cases.append(new_case)
        
        # Extract settlement amounts
        amounts = unified_ai.settlement_extractor.extract_amounts(text)
        if amounts:
            new_case['settlement_amounts'] = amounts
            new_case['max_settlement'] = max(amounts)
        
        # Extract judge if not provided
        if not judge:
            extracted_judge = unified_ai.judge_analyzer.extract_judge_name(text)
            if extracted_judge:
                new_case['judge'] = extracted_judge
        
        # Update judge statistics
        if new_case['judge']:
            unified_ai.judge_analyzer.analyze_all_judges(unified_ai.corpus.cases)
        
        # Re-create embeddings if using semantic search
        if unified_ai.embedder:
            logger.info("Updating embeddings with new case...")
            unified_ai._create_embeddings()
        
        # Save to persistent storage
        uploaded_cases_file = Path("uploaded_cases.json")
        uploaded_cases = []
        
        if uploaded_cases_file.exists():
            with open(uploaded_cases_file, 'r') as f:
                uploaded_cases = json.load(f)
        
        uploaded_cases.append(new_case)
        
        with open(uploaded_cases_file, 'w') as f:
            json.dump(uploaded_cases, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Case {citation} uploaded successfully",
            "case": new_case,
            "corpus_size": len(unified_ai.corpus.cases)
        }
        
    except Exception as e:
        logger.error(f"Case upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/cases/bulk-upload")
async def bulk_upload_cases(file: UploadFile = File(...)):
    """Upload multiple cases from a JSON or CSV file"""
    try:
        content = await file.read()
        
        if file.filename.endswith('.json'):
            cases_data = json.loads(content)
        elif file.filename.endswith('.csv'):
            import csv
            import io
            
            csv_file = io.StringIO(content.decode('utf-8'))
            reader = csv.DictReader(csv_file)
            cases_data = list(reader)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .json or .csv")
        
        if not isinstance(cases_data, list):
            raise HTTPException(status_code=400, detail="File must contain a list of cases")
        
        uploaded_count = 0
        errors = []
        
        for idx, case_data in enumerate(cases_data):
            try:
                # Validate required fields
                if not case_data.get('citation') or not case_data.get('case_name'):
                    errors.append(f"Case {idx}: Missing required fields")
                    continue
                
                # Create case object
                new_case = {
                    'citation': case_data['citation'],
                    'case_name': case_data['case_name'],
                    'text': case_data.get('text', case_data.get('judgment', ''))[:10000],
                    'outcome': case_data.get('outcome', 'unknown'),
                    'jurisdiction': case_data.get('jurisdiction', 'unknown'),
                    'year': case_data.get('year', 0),
                    'court': case_data.get('court', 'Unknown'),
                    'judge': case_data.get('judge'),
                    'catchwords': case_data.get('catchwords'),
                    'source': 'bulk_upload',
                    'uploaded_at': datetime.now().isoformat()
                }
                
                # Extract year if needed
                if not new_case['year']:
                    year_match = re.search(r'\[(\d{4})\]', new_case['citation'])
                    if year_match:
                        new_case['year'] = int(year_match.group(1))
                
                # Add to corpus
                unified_ai.corpus.cases.append(new_case)
                uploaded_count += 1
                
            except Exception as e:
                errors.append(f"Case {idx}: {str(e)}")
        
        # Update embeddings if any cases were added
        if uploaded_count > 0 and unified_ai.embedder:
            logger.info(f"Updating embeddings with {uploaded_count} new cases...")
            unified_ai._create_embeddings()
        
        # Save uploaded cases
        if uploaded_count > 0:
            uploaded_cases_file = Path("uploaded_cases.json")
            all_uploaded = []
            
            if uploaded_cases_file.exists():
                with open(uploaded_cases_file, 'r') as f:
                    all_uploaded = json.load(f)
            
            all_uploaded.extend([c for c in unified_ai.corpus.cases[-uploaded_count:]])
            
            with open(uploaded_cases_file, 'w') as f:
                json.dump(all_uploaded, f, indent=2)
        
        return {
            "status": "success",
            "uploaded": uploaded_count,
            "errors": errors,
            "corpus_size": len(unified_ai.corpus.cases)
        }
        
    except Exception as e:
        logger.error(f"Bulk upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cases/uploaded")
async def get_uploaded_cases(limit: int = 50):
    """Get recently uploaded cases"""
    try:
        uploaded_cases_file = Path("uploaded_cases.json")
        
        if uploaded_cases_file.exists():
            with open(uploaded_cases_file, 'r') as f:
                uploaded_cases = json.load(f)
            
            # Sort by upload date
            uploaded_cases.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
            
            return {
                "total": len(uploaded_cases),
                "cases": uploaded_cases[:limit]
            }
        else:
            return {
                "total": 0,
                "cases": []
            }
            
    except Exception as e:
        logger.error(f"Error retrieving uploaded cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/cases/{citation}")
async def delete_case(citation: str):
    """Delete a case from the corpus"""
    try:
        # Find and remove the case
        case_found = False
        for i, case in enumerate(unified_ai.corpus.cases):
            if case['citation'] == citation:
                unified_ai.corpus.cases.pop(i)
                case_found = True
                break
        
        if not case_found:
            raise HTTPException(status_code=404, detail=f"Case {citation} not found")
        
        # Update embeddings
        if unified_ai.embedder:
            logger.info("Updating embeddings after case deletion...")
            unified_ai._create_embeddings()
        
        # Update uploaded cases file
        uploaded_cases_file = Path("uploaded_cases.json")
        if uploaded_cases_file.exists():
            with open(uploaded_cases_file, 'r') as f:
                uploaded_cases = json.load(f)
            
            uploaded_cases = [c for c in uploaded_cases if c['citation'] != citation]
            
            with open(uploaded_cases_file, 'w') as f:
                json.dump(uploaded_cases, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Case {citation} deleted",
            "corpus_size": len(unified_ai.corpus.cases)
        }
        
    except Exception as e:
        logger.error(f"Case deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
''')
