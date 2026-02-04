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
    APP_VERSION = "2.0.0"

    # Database Configuration
    DATABASE_URL = str(DATA_DIR / "hajj_data.db")

    # Financial Configuration
    CURRENT_HAJJ_COST = 120_000_000  # IDR
    DEFAULT_INFLATION_RATE = 0.05

    # Sustainability Thresholds
    SUSTAINABILITY_CRITICAL = 40
    SUSTAINABILITY_WARNING = 60
    SUSTAINABILITY_HEALTHY = 80

    # Investment Targets
    TARGET_ROI = 0.07  # 7% target return
    TARGET_ROI_MIN = 0.05  # 5% minimum acceptable return

    # Risk Thresholds
    RISK_LOW = 20
    RISK_MEDIUM = 40
    RISK_HIGH = 60

    # Cost Thresholds
    COST_GROWTH_WARNING = 0.10  # 10% annual growth triggers warning

    # Simulation Defaults
    MAX_SIMULATIONS = 10000
    DEFAULT_SIMULATION_COUNT = 1000
    DEFAULT_TIME_HORIZON = 10
    DEFAULT_CONFIDENCE_LEVEL = 0.95

    # Projection Defaults
    DEFAULT_PROJECTION_YEARS = 10
    MAX_PROJECTION_YEARS = 25

    # Personal Planning - Shariah Returns
    SHARIAH_RETURN_CONSERVATIVE = 0.06  # 6% conservative
    SHARIAH_RETURN_MODERATE = 0.08  # 8% moderate
    SHARIAH_RETURN_AGGRESSIVE = 0.10  # 10% aggressive

    # ESG Scoring Thresholds
    ESG_EXCELLENT = 80
    ESG_GOOD = 70
    ESG_FAIR = 60

    # Chart Colors
    COLORS = {
        'bpih': '#e74c3c',
        'bipih': '#3498db',
        'sustainability': '#f39c12',
        'benefit': '#27ae60',
        'primary': '#3498db',
        'secondary': '#9b59b6',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#17a2b8'
    }
