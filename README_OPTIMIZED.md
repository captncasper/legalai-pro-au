# Optimized Legal AI API

## Quick Start

1. **Run the API:**
   ```bash
   ./run_optimized_api.sh
   ```

2. **Test the API:**
   ```bash
   python test_api.py
   ```

3. **View API docs:**
   Open http://localhost:8000/docs

## Features

- ✅ All endpoints consolidated in one file
- ✅ No module import issues
- ✅ Quantum analysis
- ✅ Monte Carlo simulations
- ✅ Strategy generation
- ✅ Case search
- ✅ WebSocket support

## Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /api/v1/analysis/quantum` - Quantum success prediction
- `POST /api/v1/prediction/simulate` - Monte Carlo simulation
- `POST /api/v1/strategy/generate` - Strategy generation
- `POST /api/v1/search/cases` - Search cases
- `WS /ws/assistant` - Real-time assistant

## Customization

Edit `optimized_main.py` to:
- Add your real LegalRAG implementation
- Connect to your actual database
- Implement real AI services
- Add authentication
