@echo off
echo � Hajj Financial Sustainability Application
echo ==========================================

echo � Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo � Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

echo �️ Initializing database...
python init_database.py
if %errorlevel% neq 0 (
    echo ⚠️ Database initialization failed, using sample data
)

echo � Starting application...
echo � Application will open at: http://localhost:8501
echo � Press Ctrl+C to stop

cd app
streamlit run main.py --server.port=8501

pause
