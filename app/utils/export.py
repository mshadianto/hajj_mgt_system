# -*- coding: utf-8 -*-
"""
Export utilities for Hajj Financial Sustainability Application

This module provides functions for exporting data to various formats
including CSV, Excel, and generating reports.
"""

import pandas as pd
import io
from datetime import datetime
from typing import Dict, List, Any, Optional


def export_to_csv(df: pd.DataFrame, filename_prefix: str = 'export') -> bytes:
    """
    Export DataFrame to CSV bytes for Streamlit download.

    Args:
        df: DataFrame to export
        filename_prefix: Prefix for the filename

    Returns:
        CSV data as bytes
    """
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding='utf-8-sig')
    return buffer.getvalue()


def export_to_excel(
    dfs: Dict[str, pd.DataFrame],
    filename_prefix: str = 'export'
) -> bytes:
    """
    Export multiple DataFrames to Excel with multiple sheets.

    Args:
        dfs: Dictionary of {sheet_name: DataFrame}
        filename_prefix: Prefix for the filename

    Returns:
        Excel data as bytes
    """
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for sheet_name, df in dfs.items():
            # Truncate sheet name to 31 chars (Excel limit)
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    return buffer.getvalue()


def generate_report_markdown(
    title: str,
    sections: Dict[str, Any],
    include_timestamp: bool = True
) -> str:
    """
    Generate a markdown report from sections.

    Args:
        title: Report title
        sections: Dictionary of {section_title: content}
        include_timestamp: Whether to include generation timestamp

    Returns:
        Markdown formatted report string
    """
    lines = [f"# {title}", ""]

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.extend([f"*Generated: {timestamp}*", ""])

    for section_title, content in sections.items():
        lines.append(f"## {section_title}")
        lines.append("")

        if isinstance(content, pd.DataFrame):
            lines.append(content.to_markdown(index=False))
        elif isinstance(content, dict):
            for key, value in content.items():
                lines.append(f"- **{key}**: {value}")
        elif isinstance(content, list):
            for item in content:
                lines.append(f"- {item}")
        else:
            lines.append(str(content))

        lines.append("")

    return "\n".join(lines)


def generate_dashboard_report(
    kpis: Dict[str, Any],
    historical_data: pd.DataFrame,
    projections: Optional[pd.DataFrame] = None,
    alerts: Optional[List[str]] = None
) -> str:
    """
    Generate a comprehensive dashboard report.

    Args:
        kpis: Dictionary of KPI metrics
        historical_data: Historical financial data
        projections: Optional projection data
        alerts: Optional list of alerts

    Returns:
        Markdown formatted dashboard report
    """
    sections = {
        "Key Performance Indicators": kpis,
        "Historical Data Summary": {
            "Years Covered": f"{historical_data['Year'].min()} - {historical_data['Year'].max()}",
            "Total Records": len(historical_data),
            "Latest BPIH": f"Rp {historical_data['BPIH'].iloc[-1]:,.0f}",
            "Latest Sustainability": f"{historical_data.get('Sustainability_Index', historical_data.get('NilaiManfaat', 0)).iloc[-1]:.1f}%"
        }
    }

    if projections is not None:
        sections["Projections Summary"] = projections.describe().to_dict()

    if alerts:
        sections["Active Alerts"] = alerts

    return generate_report_markdown(
        "Hajj Financial Sustainability Dashboard Report",
        sections
    )


def generate_esg_report(
    esg_scores: Dict[str, Any],
    islamic_compliance: Dict[str, Any],
    sustainability_metrics: Dict[str, Any]
) -> str:
    """
    Generate ESG sustainability report.

    Args:
        esg_scores: ESG scoring results
        islamic_compliance: Islamic compliance assessment
        sustainability_metrics: Sustainability metrics

    Returns:
        Markdown formatted ESG report
    """
    sections = {
        "ESG Scores Overview": {
            "Environmental Score": f"{esg_scores.get('environmental', {}).get('score', 0):.1f}",
            "Social Score": f"{esg_scores.get('social', {}).get('score', 0):.1f}",
            "Governance Score": f"{esg_scores.get('governance', {}).get('score', 0):.1f}",
            "Overall Grade": esg_scores.get('overall', {}).get('grade', 'N/A')
        },
        "Islamic Compliance": {
            "Overall Score": f"{islamic_compliance.get('overall_score', 0):.1f}",
            "Certification Status": islamic_compliance.get('certification_status', 'N/A'),
            "Portfolio Compliant": "Yes" if islamic_compliance.get('portfolio_compliant', False) else "No"
        },
        "Sustainability Metrics": {
            "Current Sustainability Index": f"{sustainability_metrics.get('current_sustainability_index', 0):.1f}%",
            "10-Year Projection": f"{sustainability_metrics.get('projected_10_year', 0):.1f}%",
            "Trend": "Improving" if sustainability_metrics.get('sustainability_trend', 0) > 0 else "Declining"
        }
    }

    return generate_report_markdown(
        "ESG & Sustainability Assessment Report",
        sections
    )


def generate_planning_report(
    user_profile: Dict[str, Any],
    hajj_calculation: Dict[str, Any],
    savings_plan: Dict[str, Any],
    recommendations: List[str]
) -> str:
    """
    Generate personal Hajj planning report.

    Args:
        user_profile: User profile information
        hajj_calculation: Hajj cost calculations
        savings_plan: Savings plan details
        recommendations: List of recommendations

    Returns:
        Markdown formatted planning report
    """
    sections = {
        "Personal Profile": user_profile,
        "Hajj Cost Calculation": hajj_calculation,
        "Savings Plan": savings_plan,
        "Recommendations": recommendations
    }

    return generate_report_markdown(
        "Personal Hajj Planning Report",
        sections
    )


def get_export_filename(prefix: str, extension: str) -> str:
    """
    Generate timestamped export filename.

    Args:
        prefix: Filename prefix
        extension: File extension (without dot)

    Returns:
        Filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"
