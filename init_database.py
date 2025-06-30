#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database initialization script for Hajj Financial Sustainability Application
Fixed encoding issues for Windows compatibility
"""

import sys
import os
from pathlib import Path
import pandas as pd
import sqlite3

def create_database_directory():
    """Create data directory if it doesn't exist"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"‚úÖ Data directory created: {data_dir}")

def create_database():
    """Create SQLite database with tables"""
    
    db_path = "data/hajj_data.db"
    
    # Database schema
    schema_sql = """
    -- Historical financial data table
    CREATE TABLE IF NOT EXISTS historical_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER NOT NULL,
        bpih REAL NOT NULL,
        bipih REAL NOT NULL,
        nilai_manfaat REAL NOT NULL,
        total_cost REAL,
        sustainability_index REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- User profiles table
    CREATE TABLE IF NOT EXISTS user_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT UNIQUE NOT NULL,
        age INTEGER,
        monthly_income REAL,
        monthly_expenses REAL,
        dependents INTEGER,
        risk_profile TEXT,
        target_hajj_year INTEGER,
        current_savings REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- ESG scores table
    CREATE TABLE IF NOT EXISTS esg_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER NOT NULL,
        environmental_score REAL,
        social_score REAL,
        governance_score REAL,
        overall_score REAL,
        islamic_compliance_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_historical_data_year ON historical_data(year);
    CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
    CREATE INDEX IF NOT EXISTS idx_esg_scores_year ON esg_scores(year);
    """
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(schema_sql)
            print(f"‚úÖ Database created successfully: {db_path}")
        return True
    except Exception as e:
        print(f"‚ùå Error creating database: {e}")
        return False

def insert_sample_data():
    """Insert sample data into database"""
    
    db_path = "data/hajj_data.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            
            # Insert historical data
            historical_data = [
                (2022, 85452883, 39886009, 45566874, 125338892, 53.3),
                (2023, 89629474, 49812700, 39816774, 139442174, 44.4),
                (2024, 94482028, 56046172, 38435856, 150528200, 40.7),
                (2025, 91493896, 60559399, 30934497, 152053295, 33.8)
            ]
            
            conn.executemany("""
                INSERT OR REPLACE INTO historical_data 
                (year, bpih, bipih, nilai_manfaat, total_cost, sustainability_index)
                VALUES (?, ?, ?, ?, ?, ?)
            """, historical_data)
            
            print(f"‚úÖ Inserted {len(historical_data)} historical records")
            
            # Insert sample user profile
            user_data = [
                ('demo_user_001', 35, 8000000.0, 6000000.0, 2, 'moderate', 2030, 15000000.0)
            ]
            
            conn.executemany("""
                INSERT OR REPLACE INTO user_profiles 
                (user_id, age, monthly_income, monthly_expenses, dependents, risk_profile, target_hajj_year, current_savings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, user_data)
            
            print("‚úÖ Created sample user profile")
            
            # Insert ESG scores
            esg_data = [
                (2022, 75.0, 85.0, 88.0, 82.7, 95.0),
                (2023, 78.0, 87.0, 90.0, 85.0, 96.0),
                (2024, 80.0, 89.0, 92.0, 87.0, 97.0)
            ]
            
            conn.executemany("""
                INSERT OR REPLACE INTO esg_scores 
                (year, environmental_score, social_score, governance_score, overall_score, islamic_compliance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, esg_data)
            
            print(f"‚úÖ Created {len(esg_data)} ESG score records")
            
        return True
        
    except Exception as e:
        print(f"‚ùå Error inserting sample data: {e}")
        return False

def verify_database():
    """Verify database was created correctly"""
    
    db_path = "data/hajj_data.db"
    
    if not os.path.exists(db_path):
        print("‚ùå Database file not found")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print(f"‚úÖ Found {len(tables)} tables: {[table[0] for table in tables]}")
            
            # Check data
            cursor.execute("SELECT COUNT(*) FROM historical_data;")
            historical_count = cursor.fetchone()[0]
            print(f"‚úÖ Historical data records: {historical_count}")
            
            cursor.execute("SELECT COUNT(*) FROM user_profiles;")
            user_count = cursor.fetchone()[0]
            print(f"‚úÖ User profile records: {user_count}")
            
            cursor.execute("SELECT COUNT(*) FROM esg_scores;")
            esg_count = cursor.fetchone()[0]
            print(f"‚úÖ ESG score records: {esg_count}")
            
        return True
        
    except Exception as e:
        print(f"‚ùå Database verification failed: {e}")
        return False

def main():
    """Main initialization function"""
    
    print("Ì∑ÑÔ∏è Initializing Hajj Financial Sustainability Database...")
    print("=" * 60)
    
    success = True
    
    # Step 1: Create directories
    print("\nÌ≥Å Creating directories...")
    create_database_directory()
    
    # Step 2: Create database
    print("\nÌ∑ÑÔ∏è Creating database...")
    if not create_database():
        success = False
    
    # Step 3: Insert sample data
    print("\nÌ≥ä Inserting sample data...")
    if not insert_sample_data():
        success = False
    
    # Step 4: Verify database
    print("\nÌ¥ç Verifying database...")
    if not verify_database():
        success = False
    
    # Final result
    print("\n" + "=" * 60)
    if success:
        print("Ìæâ Database initialization completed successfully!")
    else:
        print("‚ùå Database initialization failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
