# -*- coding: utf-8 -*-
"""
Shared utility functions for Hajj Financial Sustainability Application

This module consolidates common functions used across multiple pages
to eliminate code duplication and ensure consistency.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

# Import settings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import AppConfig


def get_grade(score: float, scale: str = 'letter') -> str:
    """
    Convert numerical score to grade.

    Args:
        score: Numerical score (0-100)
        scale: 'letter' for A+/A/A-/B+... or 'descriptive' for Excellent/Good/Fair...

    Returns:
        Grade string based on scale
    """
    if scale == 'descriptive':
        if score >= 90:
            return 'Excellent'
        elif score >= 80:
            return 'Good'
        elif score >= 70:
            return 'Fair'
        elif score >= 60:
            return 'Needs Improvement'
        else:
            return 'Poor'
    else:  # letter grade
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        else:
            return 'D'


def format_currency(value: float, currency: str = 'Rp', decimals: int = 0) -> str:
    """
    Format number as Indonesian Rupiah currency.

    Args:
        value: Numeric value to format
        currency: Currency prefix (default: 'Rp')
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    if value >= 1_000_000_000_000:
        return f"{currency} {value/1_000_000_000_000:,.{decimals}f}T"
    elif value >= 1_000_000_000:
        return f"{currency} {value/1_000_000_000:,.{decimals}f}B"
    elif value >= 1_000_000:
        return f"{currency} {value/1_000_000:,.{decimals}f}M"
    elif value >= 1_000:
        return f"{currency} {value/1_000:,.{decimals}f}K"
    else:
        return f"{currency} {value:,.{decimals}f}"


def get_dashboard_colors() -> Dict[str, str]:
    """
    Get standard dashboard color palette.

    Returns:
        Dictionary of color names to hex values
    """
    return {
        # Primary metrics
        'bpih': '#e74c3c',
        'bipih': '#3498db',
        'sustainability': '#f39c12',
        'benefit': '#27ae60',

        # Status colors
        'primary': '#3498db',
        'secondary': '#9b59b6',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#17a2b8',

        # Chart colors
        'chart_primary': '#667eea',
        'chart_secondary': '#764ba2',
        'chart_tertiary': '#00b894',

        # Gradient endpoints
        'gradient_start': '#667eea',
        'gradient_end': '#764ba2',

        # ESG colors
        'environmental': '#11998e',
        'social': '#38ef7d',
        'governance': '#667eea'
    }


def calculate_sustainability_index(benefit: float, bpih: float) -> float:
    """
    Calculate sustainability index from benefit and BPIH values.

    Args:
        benefit: NilaiManfaat (benefit value)
        bpih: BPIH (Biaya Penyelenggaraan Ibadah Haji)

    Returns:
        Sustainability index as percentage (0-100+)
    """
    if bpih <= 0:
        return 0.0
    return (benefit / bpih) * 100


def get_threshold_status(
    value: float,
    critical: float = None,
    warning: float = None,
    healthy: float = None
) -> Tuple[str, str, str]:
    """
    Determine status based on thresholds.

    Args:
        value: Current value to evaluate
        critical: Critical threshold (default from settings)
        warning: Warning threshold (default from settings)
        healthy: Healthy threshold (default from settings)

    Returns:
        Tuple of (status_text, status_class, status_emoji)
    """
    # Use defaults from settings if not provided
    critical = critical or AppConfig.SUSTAINABILITY_CRITICAL
    warning = warning or AppConfig.SUSTAINABILITY_WARNING
    healthy = healthy or AppConfig.SUSTAINABILITY_HEALTHY

    if value >= healthy:
        return ('Excellent', 'status-excellent', '🟢')
    elif value >= warning:
        return ('Good', 'status-good', '🟡')
    elif value >= critical:
        return ('Warning', 'status-warning', '🟠')
    else:
        return ('Critical', 'status-critical', '🔴')


def get_score_class(score: float) -> str:
    """
    Get CSS class name based on score value.

    Args:
        score: Numerical score (0-100)

    Returns:
        CSS class name for styling
    """
    if score >= 80:
        return 'score-excellent'
    elif score >= 70:
        return 'score-good'
    elif score >= 60:
        return 'score-fair'
    else:
        return 'score-poor'


def get_trend_indicator(current: float, previous: float, threshold: float = 0.01) -> Tuple[str, str, str]:
    """
    Calculate trend indicator between two values.

    Args:
        current: Current value
        previous: Previous value
        threshold: Minimum change to show trend (default 1%)

    Returns:
        Tuple of (trend_text, trend_class, trend_arrow)
    """
    if previous == 0:
        return ('N/A', 'trend-stable', '→')

    change = (current - previous) / previous

    if change > threshold:
        return (f'+{change*100:.1f}%', 'trend-up', '↗')
    elif change < -threshold:
        return (f'{change*100:.1f}%', 'trend-down', '↘')
    else:
        return (f'{change*100:.1f}%', 'trend-stable', '→')


def calculate_risk_level(sustainability_index: float, base_risk: float = 50) -> int:
    """
    Calculate risk level based on sustainability index.

    Args:
        sustainability_index: Current sustainability index
        base_risk: Base risk value (default 50)

    Returns:
        Risk level (0-100)
    """
    risk = base_risk - sustainability_index
    return max(0, min(100, int(risk + np.random.uniform(0, 10))))


def validate_dataframe_columns(df, required: list, optional: list = None) -> Tuple[bool, list]:
    """
    Validate that a DataFrame has required columns.

    Args:
        df: pandas DataFrame to validate
        required: List of required column names
        optional: List of optional column names

    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = [col for col in required if col not in df.columns]
    return (len(missing) == 0, missing)
