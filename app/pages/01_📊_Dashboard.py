# ============================================================================
# PAGES/01_📊_Dashboard.py - FIXED VERSION
# Solved KeyError: 'color' issue
# ============================================================================

"""
📊 MAIN DASHBOARD - FIXED
Executive dashboard with key metrics and overview
All color key errors resolved
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="📊 Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
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
    
    .scenario-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    
    .projection-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="dashboard-header">
    <h1>📊 EXECUTIVE DASHBOARD</h1>
    <h3>Hajj Financial Sustainability Overview</h3>
    <p>Real-time monitoring of key financial metrics and performance indicators</p>
</div>
""", unsafe_allow_html=True)

# FIXED: Color definitions function
def get_dashboard_colors():
    """Get standard dashboard colors - SAFE"""
    return {
        'bpih': '#e74c3c',
        'bipih': '#3498db', 
        'sustainability': '#f39c12',
        'benefit': '#27ae60',
        'primary': '#3498db',
        'secondary': '#e74c3c',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c'
    }

# Load sample data
@st.cache_data
def load_dashboard_data():
    """Load comprehensive dashboard data"""
    np.random.seed(42)
    
    # Historical data
    years = list(range(2020, 2026))
    data = []
    
    base_bpih = 75000000
    base_bipih = 35000000
    base_benefit = 50000000
    
    for i, year in enumerate(years):
        economic_cycle = np.sin(i * 0.5) * 0.02
        
        bpih = base_bpih * (1.05 + economic_cycle) ** i * np.random.uniform(0.95, 1.05)
        bipih = base_bipih * (1.07 + economic_cycle) ** i * np.random.uniform(0.9, 1.1)
        benefit = base_benefit * (0.98 + economic_cycle * 0.5) ** i * np.random.uniform(0.95, 1.05)
        
        sustainability_index = (benefit / bpih) * 100
        
        data.append({
            'Year': year,
            'BPIH': bpih,
            'Bipih': bipih,
            'NilaiManfaat': benefit,
            'Total_Cost': bpih + bipih,
            'Sustainability_Index': sustainability_index,
            'Investment_Return': 0.06 + np.random.normal(0, 0.02),
            'Risk_Score': max(0, 50 - sustainability_index) + np.random.uniform(0, 10)
        })
    
    return pd.DataFrame(data)

# Load data
df = load_dashboard_data()
current_data = df.iloc[-1]
previous_data = df.iloc[-2]
colors = get_dashboard_colors()

# Calculate trends
def get_trend(current, previous):
    change = ((current - previous) / previous) * 100
    if abs(change) < 1:
        return "stable", "→", "trend-stable"
    elif change > 0:
        return "increasing", "↗", "trend-up"
    else:
        return "decreasing", "↘", "trend-down"

# Key Performance Indicators
st.markdown("## 🎯 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    sustainability = current_data['Sustainability_Index']
    prev_sustainability = previous_data['Sustainability_Index']
    trend_text, trend_icon, trend_class = get_trend(sustainability, prev_sustainability)
    
    if sustainability >= 70:
        status_class = "status-excellent"
        status_text = "Excellent"
    elif sustainability >= 50:
        status_class = "status-good"  
        status_text = "Good"
    elif sustainability >= 30:
        status_class = "status-warning"
        status_text = "Warning"
    else:
        status_class = "status-critical"
        status_text = "Critical"
    
    st.markdown(f"""
    <div class="kpi-card {status_class}">
        <div class="kpi-label">Sustainability Index</div>
        <div class="kpi-value">{sustainability:.1f}%</div>
        <div>Status: {status_text} <span class="trend-indicator {trend_class}">{trend_icon} {abs(sustainability - prev_sustainability):.1f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_cost = current_data['Total_Cost']
    prev_cost = previous_data['Total_Cost']
    cost_change = ((total_cost - prev_cost) / prev_cost) * 100
    trend_text, trend_icon, trend_class = get_trend(total_cost, prev_cost)
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Cost</div>
        <div class="kpi-value">Rp {total_cost/1e9:.1f}B</div>
        <div>Change: <span class="trend-indicator {trend_class}">{trend_icon} {abs(cost_change):.1f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    investment_return = current_data['Investment_Return'] * 100
    target_return = 7.0
    
    if investment_return >= target_return:
        status_class = "status-excellent"
        status_text = "Above Target"
    elif investment_return >= target_return * 0.8:
        status_class = "status-good"
        status_text = "Near Target"
    else:
        status_class = "status-warning"
        status_text = "Below Target"
    
    st.markdown(f"""
    <div class="kpi-card {status_class}">
        <div class="kpi-label">Investment Return</div>
        <div class="kpi-value">{investment_return:.1f}%</div>
        <div>Target: {target_return:.1f}% ({status_text})</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    risk_score = current_data['Risk_Score']
    
    if risk_score <= 20:
        status_class = "status-excellent"
        risk_level = "Low"
    elif risk_score <= 40:
        status_class = "status-good"
        risk_level = "Medium"
    elif risk_score <= 60:
        status_class = "status-warning"
        risk_level = "High"
    else:
        status_class = "status-critical"
        risk_level = "Critical"
    
    st.markdown(f"""
    <div class="kpi-card {status_class}">
        <div class="kpi-label">Risk Level</div>
        <div class="kpi-value">{risk_score:.0f}</div>
        <div>Level: {risk_level}</div>
    </div>
    """, unsafe_allow_html=True)

# Financial Overview Charts
st.markdown("## 📈 Financial Overview")

col1, col2 = st.columns(2)

with col1:
    # Financial trend chart - FIXED COLORS
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=df['Year'],
        y=df['BPIH'],
        mode='lines+markers',
        name='BPIH',
        line=dict(color=colors['bpih'], width=3)  # FIXED: using colors dict
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=df['Year'],
        y=df['Bipih'],
        mode='lines+markers',
        name='Bipih',
        line=dict(color=colors['bipih'], width=3)  # FIXED: using colors dict
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=df['Year'],
        y=df['NilaiManfaat'],
        mode='lines+markers',
        name='Nilai Manfaat',
        line=dict(color=colors['benefit'], width=3)  # FIXED: using colors dict
    ))
    
    fig_trend.update_layout(
        title="Financial Components Trend",
        xaxis_title="Year",
        yaxis_title="Amount (IDR)",
        template="plotly_white",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    # Sustainability trend with targets - FIXED COLORS
    fig_sustainability = go.Figure()
    
    fig_sustainability.add_trace(go.Scatter(
        x=df['Year'],
        y=df['Sustainability_Index'],
        mode='lines+markers',
        name='Sustainability Index',
        line=dict(color=colors['sustainability'], width=4),  # FIXED: using colors dict
        fill='tonexty'
    ))
    
    # Add threshold lines
    fig_sustainability.add_hline(
        y=70, line_dash="dash", line_color=colors['success'],
        annotation_text="Excellent (70%)"
    )
    fig_sustainability.add_hline(
        y=50, line_dash="dash", line_color=colors['warning'],
        annotation_text="Good (50%)"
    )
    fig_sustainability.add_hline(
        y=30, line_dash="dash", line_color=colors['danger'],
        annotation_text="Critical (30%)"
    )
    
    fig_sustainability.update_layout(
        title="Sustainability Index Trend",
        xaxis_title="Year",
        yaxis_title="Sustainability Index (%)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_sustainability, use_container_width=True)

# Performance Summary
st.markdown("## 📊 Performance Summary")

col1, col2, col3 = st.columns(3)

with col1:
    # Year-over-year changes
    yoy_data = []
    
    metrics = ['BPIH', 'Bipih', 'NilaiManfaat', 'Sustainability_Index']
    
    for metric in metrics:
        current_val = current_data[metric]
        previous_val = previous_data[metric]
        change = ((current_val - previous_val) / previous_val) * 100
        
        yoy_data.append({
            'Metric': metric.replace('_', ' '),
            'Current': f"Rp {current_val:,.0f}" if 'Index' not in metric else f"{current_val:.1f}%",
            'YoY Change': f"{change:+.1f}%"
        })
    
    yoy_df = pd.DataFrame(yoy_data)
    st.markdown("### 📅 Year-over-Year Changes")
    st.dataframe(yoy_df, use_container_width=True)

with col2:
    # Risk assessment
    st.markdown("### ⚠️ Risk Assessment")
    
    risk_factors = [
        ("Market Risk", np.random.uniform(20, 80)),
        ("Operational Risk", np.random.uniform(10, 50)),
        ("Regulatory Risk", np.random.uniform(5, 30)),
        ("Liquidity Risk", np.random.uniform(15, 60))
    ]
    
    risk_df = pd.DataFrame(risk_factors, columns=['Risk Type', 'Score'])
    risk_df['Level'] = risk_df['Score'].apply(
        lambda x: 'Low' if x < 30 else 'Medium' if x < 60 else 'High'
    )
    
    fig_risk = px.bar(
        risk_df,
        x='Score',
        y='Risk Type',
        orientation='h',
        color='Level',
        color_discrete_map={'Low': colors['success'], 'Medium': colors['warning'], 'High': colors['danger']},
        title="Risk Factors Assessment"
    )
    fig_risk.update_layout(height=300)
    st.plotly_chart(fig_risk, use_container_width=True)

with col3:
    # Portfolio allocation
    st.markdown("### 💼 Portfolio Allocation")
    
    allocation_data = {
        'Asset Class': ['Sukuk', 'Equity', 'Real Estate', 'Cash', 'Commodities'],
        'Allocation': [40, 30, 15, 10, 5],
        'Performance': [6.2, 8.5, 7.1, 3.5, 4.8]
    }
    
    allocation_df = pd.DataFrame(allocation_data)
    
    fig_allocation = px.pie(
        allocation_df,
        values='Allocation',
        names='Asset Class',
        title="Asset Allocation",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_allocation.update_layout(height=300)
    st.plotly_chart(fig_allocation, use_container_width=True)

# FIXED: Scenario Analysis Section
st.markdown("## 🎯 Scenario Analysis")

# FIXED: Define scenario parameters with proper color structure
scenario_params = {
    'pessimistic': {
        'name': 'Pessimistic Scenario',
        'description': 'Conservative assumptions with higher costs and lower returns',
        'color': colors['danger'],  # FIXED: color at top level
        'adjustments': {
            'BPIH': 1.15,  # 15% higher costs
            'Bipih': 1.12,  # 12% higher costs  
            'NilaiManfaat': 0.90  # 10% lower returns
        }
    },
    'baseline': {
        'name': 'Baseline Scenario',
        'description': 'Current trend continuation with moderate assumptions',
        'color': colors['primary'],  # FIXED: color at top level
        'adjustments': {
            'BPIH': 1.0,
            'Bipih': 1.0,
            'NilaiManfaat': 1.0
        }
    },
    'optimistic': {
        'name': 'Optimistic Scenario', 
        'description': 'Favorable conditions with cost efficiency and higher returns',
        'color': colors['success'],  # FIXED: color at top level
        'adjustments': {
            'BPIH': 0.95,  # 5% cost reduction
            'Bipih': 0.93,  # 7% cost reduction
            'NilaiManfaat': 1.15  # 15% higher returns
        }
    }
}

# Run quick scenario projections
projection_years = 5
future_years = list(range(df['Year'].max() + 1, df['Year'].max() + projection_years + 1))

# Simple linear projections for scenarios
scenario_results = {}

for scenario_name, params in scenario_params.items():
    # Calculate simple growth projections
    current_bpih = current_data['BPIH']
    current_benefit = current_data['NilaiManfaat']
    
    # Apply scenario adjustments
    adjusted_bpih = current_bpih * params['adjustments']['BPIH']
    adjusted_benefit = current_benefit * params['adjustments']['NilaiManfaat']
    
    # Simple linear projection
    bpih_growth = 0.05  # 5% annual growth base
    benefit_growth = -0.02  # 2% annual decline base
    
    projected_sustainability = []
    for year in range(projection_years):
        future_bpih = adjusted_bpih * (1 + bpih_growth) ** (year + 1)
        future_benefit = adjusted_benefit * (1 + benefit_growth) ** (year + 1)
        sustainability = (future_benefit / future_bpih) * 100
        projected_sustainability.append(sustainability)
    
    scenario_results[scenario_name] = {
        'sustainability': projected_sustainability,
        'params': params
    }

# Scenario comparison visualization
fig_scenarios = go.Figure()

for scenario_name, scenario_data in scenario_results.items():
    params = scenario_data['params']
    
    fig_scenarios.add_trace(go.Scatter(
        x=future_years,
        y=scenario_data['sustainability'],
        mode='lines+markers',
        name=params['name'],
        line=dict(color=params['color'], width=3)  # FIXED: using correct color access
    ))

# Add threshold lines
fig_scenarios.add_hline(y=70, line_dash="dot", line_color=colors['success'], annotation_text="Excellent (70%)")
fig_scenarios.add_hline(y=50, line_dash="dot", line_color=colors['warning'], annotation_text="Good (50%)")
fig_scenarios.add_hline(y=30, line_dash="dot", line_color=colors['danger'], annotation_text="Critical (30%)")

fig_scenarios.update_layout(
    title="Sustainability Index - Scenario Analysis",
    xaxis_title="Year",
    yaxis_title="Sustainability Index (%)",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_scenarios, use_container_width=True)

# FIXED: Scenario summary
st.markdown("### 📋 Scenario Summary")

col1, col2, col3 = st.columns(3)

for i, (scenario_name, scenario_data) in enumerate(scenario_results.items()):
    with [col1, col2, col3][i]:
        params = scenario_data['params']
        final_sustainability = scenario_data['sustainability'][-1]
        
        st.markdown(f"""
        <div class="scenario-card">
            <h4 style="color: {params['color']};">{params['name']}</h4>
            <p><strong>Final Sustainability:</strong> {final_sustainability:.1f}%</p>
            <p><strong>Description:</strong> {params['description']}</p>
            <p><strong>Key Assumptions:</strong></p>
            <ul>
                <li>BPIH: {(params['adjustments']['BPIH'] - 1) * 100:+.0f}%</li>
                <li>Benefit: {(params['adjustments']['NilaiManfaat'] - 1) * 100:+.0f}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Real-time Alerts
st.markdown("## 🚨 Real-time Alerts & Notifications")

# Generate sample alerts
alerts = []

if current_data['Sustainability_Index'] < 40:
    alerts.append({
        'Level': 'Critical',
        'Message': f"Sustainability index ({current_data['Sustainability_Index']:.1f}%) below critical threshold",
        'Action': 'Immediate intervention required',
        'Icon': '🚨'
    })

if current_data['Investment_Return'] < 0.05:
    alerts.append({
        'Level': 'Warning',
        'Message': f"Investment returns ({current_data['Investment_Return']:.1%}) below target",
        'Action': 'Review investment strategy',
        'Icon': '⚠️'
    })

cost_growth = ((current_data['Total_Cost'] - previous_data['Total_Cost']) / previous_data['Total_Cost']) * 100
if cost_growth > 10:
    alerts.append({
        'Level': 'Warning',
        'Message': f"Cost growth ({cost_growth:.1f}%) exceeds threshold",
        'Action': 'Implement cost control measures',
        'Icon': '📈'
    })

if not alerts:
    alerts.append({
        'Level': 'Success',
        'Message': 'All key metrics within acceptable ranges',
        'Action': 'Continue monitoring',
        'Icon': '✅'
    })

# Display alerts
for alert in alerts:
    level_colors = {
        'Critical': colors['danger'],
        'Warning': colors['warning'], 
        'Success': colors['success'],
        'Info': colors['primary']
    }
    
    color = level_colors.get(alert['Level'], colors['primary'])
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, {color}22 0%, {color}11 100%);
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    ">
        <strong>{alert['Icon']} {alert['Level']}:</strong> {alert['Message']}<br>
        <small><strong>Recommended Action:</strong> {alert['Action']}</small>
    </div>
    """, unsafe_allow_html=True)

# Quick Actions
st.markdown("## ⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Generate Report", use_container_width=True):
        st.success("📋 Executive report generated!")

with col2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("🔄 Data refreshed!")

with col3:
    if st.button("📈 Run Analysis", use_container_width=True):
        st.info("🧮 Advanced analysis initiated...")

with col4:
    if st.button("⚙️ Settings", use_container_width=True):
        st.info("⚙️ Settings panel opened")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><em>Dashboard last updated: {}</em></p>
    <p>🕌 Hajj Financial Sustainability Dashboard | Real-time Monitoring & Analytics</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)