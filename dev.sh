#!/bin/bash

# Hajj Financial Sustainability Application - Development Script

echo "Ìª†Ô∏è Starting development environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Ì≥¶ Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "ÔøΩÔøΩ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Initialize database if it doesn't exist
if [ ! -f "data/hajj_data.db" ]; then
    echo "Ì∑ÑÔ∏è Initializing database..."
    python init_database.py
fi

# Set development environment
export DEBUG=True
export LOG_LEVEL=DEBUG

# Start the application
echo "Ì∫Ä Starting Streamlit application..."
echo "Ì≥ä Application will be available at: http://localhost:8501"
echo "Ì≤° Press Ctrl+C to stop the application"

cd app
streamlit run main.py --server.port=8501
