# Ì∫Ä Quick Start Guide

## Hajj Financial Sustainability Application

### ‚ö° Super Quick Start (1 minute)

```bash
# Clone and run
git clone <repository-url>
cd hajj_sustainability_app
./dev.sh
```

Visit `http://localhost:8501` in your browser!

### Ì≥ã Prerequisites

- Python 3.8+ 
- 4GB+ RAM
- Internet connection

### Ìª†Ô∏è Development Setup

1. **Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configuration**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Database Initialization**
```bash
python init_database.py
```

4. **Run Application**
```bash
streamlit run app/main.py
```

### Ì∞≥ Docker Quick Start

```bash
# Build and run with Docker
docker build -t hajj-app .
docker run -p 8501:8501 hajj-app

# Or use Docker Compose
docker-compose up
```

### Ì≥ä Features Overview

- **Ì≥à Executive Dashboard**: Real-time KPIs and alerts
- **Ì¥Æ Financial Projections**: AI-powered forecasting
- **ÌæØ AI Optimization**: Genetic algorithms and ML
- **ÔøΩÔøΩ RAG Assistant**: Intelligent chatbot
- **Ì≥ä Advanced Analytics**: Risk and performance analysis
- **‚ö° Monte Carlo**: Stochastic simulations
- **Ìº± ESG & Sustainability**: Comprehensive ESG scoring
- **Ì≤º Personal Planning**: Individual hajj planning

### Ì∂ò Troubleshooting

**Installation Issues:**
```bash
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

**Performance Issues:**
- Reduce simulation parameters in sidebar
- Enable caching in settings
- Use Docker for better isolation

**Data Issues:**
- Check file formats (CSV/Excel)
- Verify column names match expected format
- Ensure data files are in `data/raw/` directory

### Ì≥û Support

- GitHub Issues: Bug reports and features
- Documentation: Check the full README.md
- Email: support@hajjfinance.com

---

*Ìµå Built with ‚ù§Ô∏è for sustainable hajj financing*
