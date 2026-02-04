# -*- coding: utf-8 -*-
"""
Shared CSS styles for Hajj Financial Sustainability Application

This module consolidates CSS styles used across multiple pages
to eliminate duplication and ensure consistent styling.
"""

from typing import Dict


def get_base_styles() -> str:
    """
    Get base CSS styles used across all pages.

    Returns:
        CSS string for base styles
    """
    return """
    <style>
        /* Base card styling */
        .base-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            transition: transform 0.3s ease;
        }

        .base-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }

        /* Score circles */
        .score-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            color: white;
            font-size: 24px;
            font-weight: bold;
        }

        .score-excellent {
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        }

        .score-good {
            background: linear-gradient(135deg, #00b894 0%, #55a3ff 100%);
        }

        .score-fair {
            background: linear-gradient(135deg, #fdcb6e 0%, #f39c12 100%);
        }

        .score-poor {
            background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        }

        /* Status indicators */
        .status-excellent {
            color: #27ae60;
            border-left: 4px solid #27ae60;
        }

        .status-good {
            color: #f39c12;
            border-left: 4px solid #f39c12;
        }

        .status-warning {
            color: #e67e22;
            border-left: 4px solid #e67e22;
        }

        .status-critical {
            color: #e74c3c;
            border-left: 4px solid #e74c3c;
        }

        /* Trend indicators */
        .trend-indicator {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
            margin-left: 0.5rem;
        }

        .trend-up {
            background: #d4edda;
            color: #155724;
        }

        .trend-down {
            background: #f8d7da;
            color: #721c24;
        }

        .trend-stable {
            background: #d1ecf1;
            color: #0c5460;
        }

        /* Compliance badges */
        .compliance-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            margin: 0.2rem;
        }

        .compliant {
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        }

        .non-compliant {
            background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        }

        .under-review {
            background: linear-gradient(135deg, #fdcb6e 0%, #f39c12 100%);
        }

        /* Milestone badges */
        .milestone-achieved {
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
        }

        .milestone-pending {
            background: linear-gradient(135deg, #b2bec3 0%, #636e72 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
        }
    </style>
    """


def get_header_gradient(theme: str = 'blue') -> str:
    """
    Get header gradient CSS based on theme.

    Args:
        theme: Color theme ('blue', 'purple', 'green', 'cyan', 'orange')

    Returns:
        CSS string for header gradient
    """
    gradients = {
        'blue': ('linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%)',
                 'rgba(30, 58, 138, 0.3)'),
        'purple': ('linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                   'rgba(102, 126, 234, 0.3)'),
        'green': ('linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
                  'rgba(17, 153, 142, 0.3)'),
        'cyan': ('linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                 'rgba(79, 172, 254, 0.3)'),
        'orange': ('linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                   'rgba(240, 147, 251, 0.3)')
    }

    gradient, shadow = gradients.get(theme, gradients['blue'])

    return f"""
    <style>
        .page-header {{
            background: {gradient};
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 25px {shadow};
        }}

        .page-header h1 {{
            margin: 0;
            font-size: 2.5rem;
        }}

        .page-header h3 {{
            margin: 0.5rem 0;
            opacity: 0.9;
        }}

        .page-header p {{
            margin: 0;
            opacity: 0.8;
        }}
    </style>
    """


def get_card_styles(card_type: str = 'default') -> str:
    """
    Get card-specific CSS styles.

    Args:
        card_type: Type of card ('kpi', 'esg', 'scenario', 'calculator', 'simulation')

    Returns:
        CSS string for card styles
    """
    styles = {
        'kpi': """
        <style>
            .kpi-card {
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                margin: 1rem 0;
                transition: transform 0.3s ease;
            }

            .kpi-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            }

            .kpi-value {
                font-size: 2.5rem;
                font-weight: bold;
                margin: 0.5rem 0;
            }

            .kpi-label {
                font-size: 0.9rem;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
        </style>
        """,

        'esg': """
        <style>
            .esg-card {
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin: 1rem 0;
                border-left: 4px solid #27ae60;
                transition: transform 0.3s ease;
            }

            .esg-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            }
        </style>
        """,

        'scenario': """
        <style>
            .scenario-card {
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin: 1rem 0;
                border-left: 4px solid #3498db;
            }
        </style>
        """,

        'calculator': """
        <style>
            .calculator-card {
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin: 1rem 0;
                border-left: 4px solid #667eea;
            }
        </style>
        """,

        'simulation': """
        <style>
            .simulation-card {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(168, 237, 234, 0.3);
            }
        </style>
        """,

        'result': """
        <style>
            .result-highlight {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 1.5rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
            }
        </style>
        """
    }

    return styles.get(card_type, styles['kpi'])


def get_alert_styles() -> str:
    """
    Get alert/notification CSS styles.

    Returns:
        CSS string for alert styles
    """
    return """
    <style>
        .alert-box {
            padding: 1rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .alert-success {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left: 4px solid #27ae60;
            color: #155724;
        }

        .alert-warning {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
            border-left: 4px solid #f39c12;
            color: #856404;
        }

        .alert-danger {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-left: 4px solid #e74c3c;
            color: #721c24;
        }

        .alert-info {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }

        .alert-icon {
            font-size: 1.5rem;
        }

        .alert-content {
            flex: 1;
        }

        .alert-title {
            font-weight: bold;
            margin-bottom: 0.25rem;
        }

        .alert-message {
            font-size: 0.9rem;
            opacity: 0.9;
        }
    </style>
    """


def get_all_styles() -> str:
    """
    Get all combined CSS styles.

    Returns:
        Complete CSS string for all styles
    """
    return (
        get_base_styles() +
        get_header_gradient('blue') +
        get_card_styles('kpi') +
        get_alert_styles()
    )
