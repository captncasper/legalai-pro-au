#!/usr/bin/env python3
"""
Web UI for case upload - add this endpoint to your unified system
"""

from fastapi.responses import HTMLResponse

# Add this endpoint to your unified system:

print('''
@app.get("/upload", response_class=HTMLResponse)
async def upload_ui():
    """Simple web interface for uploading cases"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Australian Legal AI - Case Upload</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #34495e;
            }
            input, textarea, select {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            textarea {
                min-height: 200px;
                resize: vertical;
            }
            button {
                background-color: #3498db;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                margin-right: 10px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .success {
                background-color: #2ecc71;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                display: none;
            }
            .error {
                background-color: #e74c3c;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                display: none;
            }
            .bulk-upload {
                margin-top: 40px;
                padding-top: 30px;
                border-top: 2px solid #ecf0f1;
            }
            .stats {
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .required {
                color: #e74c3c;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏛️ Australian Legal AI - Case Upload</h1>
            
            <div class="stats" id="stats">
                Loading corpus statistics...
            </div>
            
            <div class="success" id="success-message"></div>
            <div class="error" id="error-message"></div>
            
            <form id="upload-form">
                <h2>Upload Single Case</h2>
                
                <div class="form-group">
                    <label>Citation <span class="required">*</span></label>
                    <input type="text" name="citation" required 
                           placeholder="e.g., [2023] NSWSC 100">
                </div>
                
                <div class="form-group">
                    <label>Case Name <span class="required">*</span></label>
                    <input type="text" name="case_name" required 
                           placeholder="e.g., Smith v Jones">
                </div>
                
                <div class="form-group">
                    <label>Case Text / Judgment <span class="required">*</span></label>
                    <textarea name="text" required 
                              placeholder="Paste the full judgment text here..."></textarea>
                </div>
                
                <div class="form-group">
                    <label>Outcome</label>
                    <select name="outcome">
                        <option value="unknown">Unknown</option>
                        <option value="applicant_won">Applicant Won</option>
                        <option value="applicant_lost">Applicant Lost</option>
                        <option value="settled">Settled</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Jurisdiction</label>
                    <select name="jurisdiction">
                        <option value="nsw">New South Wales</option>
                        <option value="vic">Victoria</option>
                        <option value="qld">Queensland</option>
                        <option value="wa">Western Australia</option>
                        <option value="sa">South Australia</option>
                        <option value="tas">Tasmania</option>
                        <option value="act">ACT</option>
                        <option value="nt">Northern Territory</option>
                        <option value="federal">Federal</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Court</label>
                    <input type="text" name="court" 
                           placeholder="e.g., NSWSC, FCA, HCA">
                </div>
                
                <div class="form-group">
                    <label>Judge</label>
                    <input type="text" name="judge" 
                           placeholder="e.g., Smith J">
                </div>
                
                <div class="form-group">
                    <label>Catchwords</label>
                    <textarea name="catchwords" rows="3"
                              placeholder="e.g., CONTRACT - breach - damages - mitigation"></textarea>
                </div>
                
                <button type="submit">Upload Case</button>
                <button type="reset">Clear Form</button>
            </form>
            
            <div class="bulk-upload">
                <h2>Bulk Upload</h2>
                <form id="bulk-upload-form" enctype="multipart/form-data">
                    <div class="form-group">
                        <label>Upload JSON or CSV file</label>
                        <input type="file" name="file" accept=".json,.csv" required>
                        <p style="color: #7f8c8d; font-size: 14px;">
                            File must contain: citation, case_name, text (required) + 
                            outcome, jurisdiction, court, judge, catchwords (optional)
                        </p>
                    </div>
                    <button type="submit">Upload File</button>
                </form>
            </div>
        </div>
        
        <script>
            // Load statistics
            async function loadStats() {
                try {
                    const response = await fetch('/api/v1/statistics');
                    const data = await response.json();
                    document.getElementById('stats').innerHTML = `
                        <strong>Corpus Statistics:</strong> 
                        ${data.total_cases} cases | 
                        ${Object.keys(data.outcome_distribution).length} outcome types | 
                        ${data.jurisdictions} jurisdictions
                    `;
                } catch (error) {
                    console.error('Error loading stats:', error);
                }
            }
            
            // Single case upload
            document.getElementById('upload-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                
                try {
                    const response = await fetch('/api/v1/cases/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        document.getElementById('success-message').textContent = 
                            `✅ ${result.message}. Corpus now has ${result.corpus_size} cases.`;
                        document.getElementById('success-message').style.display = 'block';
                        document.getElementById('error-message').style.display = 'none';
                        e.target.reset();
                        loadStats();
                    } else {
                        throw new Error(result.detail || 'Upload failed');
                    }
                } catch (error) {
                    document.getElementById('error-message').textContent = 
                        `❌ Error: ${error.message}`;
                    document.getElementById('error-message').style.display = 'block';
                    document.getElementById('success-message').style.display = 'none';
                }
            });
            
            // Bulk upload
            document.getElementById('bulk-upload-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                
                try {
                    const response = await fetch('/api/v1/cases/bulk-upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        let message = `✅ Uploaded ${result.uploaded} cases successfully.`;
                        if (result.errors.length > 0) {
                            message += ` ${result.errors.length} errors occurred.`;
                        }
                        message += ` Corpus now has ${result.corpus_size} cases.`;
                        
                        document.getElementById('success-message').textContent = message;
                        document.getElementById('success-message').style.display = 'block';
                        document.getElementById('error-message').style.display = 'none';
                        e.target.reset();
                        loadStats();
                    } else {
                        throw new Error(result.detail || 'Bulk upload failed');
                    }
                } catch (error) {
                    document.getElementById('error-message').textContent = 
                        `❌ Error: ${error.message}`;
                    document.getElementById('error-message').style.display = 'block';
                    document.getElementById('success-message').style.display = 'none';
                }
            });
            
            // Load stats on page load
            loadStats();
        </script>
    </body>
    </html>
    """
''')
