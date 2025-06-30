#!/usr/bin/env python3
"""
Setup verification script for Hajj Financial Sustainability Application
"""

import sys
import os
from pathlib import Path
import importlib.util

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} is not supported. Requires Python 3.8+")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'scikit-learn',
        'scipy', 'sqlalchemy', 'langchain', 'openai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_file_structure():
    """Check if all required files exist"""
    required_files = [
        'app/main.py',
        'app/pages/01_�_Dashboard.py',
        'config/settings.py',
        'requirements.txt',
        '.env',
        'data/raw/biaya_haji_historis.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def check_database():
    """Check database initialization"""
    try:
        import sqlite3
        if os.path.exists('data/hajj_data.db'):
            conn = sqlite3.connect('data/hajj_data.db')
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if len(tables) > 0:
                print(f"✅ Database initialized with {len(tables)} tables")
                return True
            else:
                print("❌ Database exists but no tables found")
                return False
        else:
            print("❌ Database file not found")
            return False
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False

def main():
    """Main verification function"""
    print("� Verifying Hajj Financial Sustainability Application Setup...")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Check Python version
    print("\n� Checking Python version...")
    if not check_python_version():
        all_checks_passed = False
    
    # Check required packages
    print("\n� Checking required packages...")
    packages_ok, missing_packages = check_required_packages()
    if not packages_ok:
        all_checks_passed = False
        print(f"\n� Install missing packages with: pip install {' '.join(missing_packages)}")
    
    # Check file structure
    print("\n� Checking file structure...")
    files_ok, missing_files = check_file_structure()
    if not files_ok:
        all_checks_passed = False
        print(f"\n� Missing files: {', '.join(missing_files)}")
    
    # Check database
    print("\n�️ Checking database...")
    if not check_database():
        all_checks_passed = False
        print("\n� Initialize database with: python init_database.py")
    
    # Final result
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("� Setup verification PASSED! Application is ready to run.")
        print("\n� Start the application with:")
        print("   ./dev.sh")
        print("   OR")
        print("   streamlit run app/main.py")
    else:
        print("❌ Setup verification FAILED! Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
