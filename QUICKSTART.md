# � Quick Start Guide

## Hajj Financial Sustainability Application

### ⚡ Super Quick Start (1 minute)

```bash
# Clone and run
git clone <repository-url>
cd hajj_sustainability_app
./dev.sh
```

Visit `http://localhost:8501` in your browser!

### � Prerequisites

- Python 3.8+ 
- 4GB+ RAM
- Internet connection

### �️ Development Setup

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

### � Docker Quick Start

```bash
# Build and run with Docker
docker build -t hajj-app .
docker run -p 8501:8501 hajj-app

# Or use Docker Compose
docker-compose up
```

### � Features Overview

- **� Executive Dashboard**: Real-time KPIs and alerts
- **� Financial Projections**: AI-powered forecasting
- **� AI Optimization**: Genetic algorithms and ML
- **�� RAG Assistant**: Intelligent chatbot
- **� Advanced Analytics**: Risk and performance analysis
- **⚡ Monte Carlo**: Stochastic simulations
- **� ESG & Sustainability**: Comprehensive ESG scoring
- **� Personal Planning**: Individual hajj planning

### � Troubleshooting

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

### � Support

- GitHub Issues: Bug reports and features
- Documentation: Check the full README.md
- Email: support@hajjfinance.com

---

*� Built with ❤️ for sustainable hajj financing*
