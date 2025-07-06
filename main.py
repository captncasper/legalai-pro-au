from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Australian Legal AI", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/api")
def api_root():
    return {
        "message": "🏛️ Australian Legal AI is LIVE!",
        "status": "operational",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": "2024-01-01"}

@app.post("/api/v1/search")
def search(query: str = "contract law"):
    return {
        "status": "success",
        "query": query,
        "results": [
            {
                "citation": "[2023] HCA 1",
                "case_name": f"Demo case for: {query}",
                "summary": "This is a working demonstration of the Australian Legal AI search.",
                "relevance_score": 0.95
            }
        ],
        "message": "API is working! Full corpus coming soon."
    }
