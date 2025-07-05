#!/usr/bin/env python3
"""
Production startup script for Railway deployment
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_production_environment():
    """Setup production environment"""
    
    # Create necessary directories
    dirs_to_create = [
        "scraped_cases",
        "logs", 
        "rag_index",
        "models",
        "data"
    ]
    
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {dir_name}")
    
    # Set environment variables for production
    os.environ.setdefault("ENVIRONMENT", "production")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    
    logger.info("🚀 Production environment setup complete!")

if __name__ == "__main__":
    setup_production_environment()
    
    # Import and run the main app
    import uvicorn
    from unified_legal_ai_system_fixed import app
    
    # Get port from Railway environment variable
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚂 Starting Australian Legal AI on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
