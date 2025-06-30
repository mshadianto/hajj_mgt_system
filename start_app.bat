@echo off
echo Ìµå Hajj Financial Sustainability Application
echo ==========================================

echo Ì≥ã Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ‚ùå Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo Ì≥¶ Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ‚ùå Failed to install requirements
    pause
    exit /b 1
)

echo Ì∑ÑÔ∏è Initializing database...
python init_database.py
if %errorlevel% neq 0 (
    echo ‚ö†Ô∏è Database initialization failed, using sample data
)

echo Ì∫Ä Starting application...
echo Ì≥ä Application will open at: http://localhost:8501
echo Ì≤° Press Ctrl+C to stop

cd app
streamlit run main.py --server.port=8501

pause
