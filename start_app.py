#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple startup script for Hajj Financial Sustainability Application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if basic requirements are installed"""
    required_packages = ['streamlit', 'pandas', 'plotly']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"‚úÖ {package}")
        except ImportError:
            print(f"‚ùå {package} - Not installed")
            missing.append(package)
    
    return missing

def install_requirements():
    """Install requirements"""
    print("Ì≥¶ Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("‚úÖ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"‚ùå Failed to install requirements: {e}")
        return False

def initialize_database():
    """Initialize database"""
    print("Ì∑ÑÔ∏è Initializing database...")
    try:
        subprocess.check_call([sys.executable, "init_database.py"])
        print("‚úÖ Database initialized successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"‚ùå Failed to initialize database: {e}")
        return False

def start_streamlit():
    """Start Streamlit application"""
    print("Ì∫Ä Starting Streamlit application...")
    print("Ì≥ä Application will be available at: http://localhost:8501")
    print("Ì≤° Press Ctrl+C to stop the application")
    
    try:
        os.chdir("app")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("\nÌªë Application stopped by user")
    except Exception as e:
        print(f"‚ùå Error starting application: {e}")

def main():
    """Main startup function"""
    print("Ìµå Hajj Financial Sustainability Application")
    print("=" * 50)
    
    # Check requirements
    print("\nÌ≥ã Checking requirements...")
    missing = check_requirements()
    
    if missing:
        print(f"\nÌ≥¶ Installing missing packages: {', '.join(missing)}")
        if not install_requirements():
            print("‚ùå Installation failed. Please install manually:")
            print(f"   pip install {' '.join(missing)}")
            return 1
    
    # Initialize database
    if not os.path.exists("data/hajj_data.db"):
        if not initialize_database():
            print("‚ö†Ô∏è Database initialization failed. Application will use sample data.")
    
    # Start application
    start_streamlit()
    
    return 0

if __name__ == "__main__":
    exit(main())
