# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hajj Financial Sustainability Dashboard - A Streamlit-based financial analytics application for monitoring and optimizing Hajj fund management. Built with AI/ML capabilities including genetic algorithm optimization, Monte Carlo simulations, and RAG-powered chatbot.

## Development Commands

```bash
# Setup and run
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python init_database.py
streamlit run app/main.py --server.port=8501

# Quick start (Linux/macOS)
./dev.sh

# Docker
docker build -t hajj-app .
docker run -p 8501:8501 hajj-app
docker-compose up

# Testing (CI/CD uses this)
pip install pytest pytest-cov
pytest tests/ -v --cov=app
```

## Architecture

### Application Structure
- **Entry point**: `app/main.py` - Landing page and main dashboard
- **Multi-page app**: `app/pages/` - Streamlit's convention for multi-page apps (numbered/emoji prefixed)
- **Configuration**: `config/settings.py` - AppConfig class with thresholds and constants
- **Database**: SQLite at `data/hajj_data.db`, initialized via `init_database.py`

### Page Modules (app/pages/)
1. `01_📊_Dashboard.py` - Executive KPI monitoring
2. `02_🔮_Projections.py` - Financial forecasting models
3. `03_🎯_Optimization.py` - Genetic algorithm portfolio optimization
4. `04_🤖_RAG_Assistant.py` - AI chatbot with LangChain/OpenAI
5. `05_📈_Analytics.py` - Statistical analysis and insights
6. `06_⚡_Simulation.py` - Monte Carlo stochastic simulations
7. `07_🌱_Sustainability.py` - ESG scoring and Islamic finance compliance
8. `08_💼_Planning.py` - Personal hajj planning tools

### Key Technologies
- **Frontend**: Streamlit with custom CSS (`assets/css/style.css`)
- **Visualization**: Plotly (interactive), Matplotlib, Seaborn
- **ML/Optimization**: scikit-learn, custom genetic algorithms, scipy
- **RAG Stack**: LangChain + OpenAI + ChromaDB (requires OPENAI_API_KEY env var)
- **Data**: Pandas, NumPy, SQLite

### Configuration Files
- `.streamlit/config.toml` - Streamlit server settings (port 8501, light theme)
- `config/settings.py` - Application constants:
  - `CURRENT_HAJJ_COST`: 120,000,000 IDR
  - `DEFAULT_INFLATION_RATE`: 0.05
  - Sustainability thresholds: CRITICAL=40, WARNING=60, HEALTHY=80

### Data Flow
- Raw data in `data/raw/` (CSV/Excel, e.g., `portfolio_sample.json`)
- Processed data in `data/processed/`
- User profiles and historical data stored in SQLite

## Environment Variables

Required for RAG Assistant functionality:
```
OPENAI_API_KEY=your_openai_api_key
```

## Live Demo

https://hajj-mgt-system-mshadianto.streamlit.app/
