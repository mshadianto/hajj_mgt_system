# -*- coding: utf-8 -*-
"""
Application settings and configuration
"""

import os
from pathlib import Path

# Base Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

class AppConfig:
    """Application configuration class"""
    
    # Application Info
    APP_NAME = "Hajj Financial Sustainability Application"
    APP_VERSION = "1.0.0"
    
    # Database Configuration
    DATABASE_URL = str(DATA_DIR / "hajj_data.db")
    
    # Financial Configuration
    CURRENT_HAJJ_COST = 120_000_000  # IDR
    DEFAULT_INFLATION_RATE = 0.05
    
    # Sustainability Thresholds
    SUSTAINABILITY_CRITICAL = 40
    SUSTAINABILITY_WARNING = 60
    SUSTAINABILITY_HEALTHY = 80
