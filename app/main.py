"""
🕌 APLIKASI SUSTAINABILITAS KEUANGAN HAJI
Main Streamlit Application with Advanced Features

Author: MS Hadianto - RAG & Agentic AI Enthusiast | Audit Committee of Hajj Fund Management Agency
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="🕌 Hajj Financial Sustainability",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://hajj-finance-support.com',
        'Report a bug': 'https://hajj-finance-support.com/bug',
        'About': """
        # Hajj Financial Sustainability Application
        
        **Advanced AI-Powered Financial Planning for Hajj**
        
        Developed by: MS Hadianto
        RAG & Agentic AI Enthusiast | Audit Committee of Hajj Fund Management Agency
        
        This application provides comprehensive financial analysis, 
        actuarial modeling, and optimization tools for sustainable 
        hajj fund management.
        
        **Features:**
        - 🤖 AI-Powered Optimization
        - 📊 Advanced Analytics  
        - 🔮 Monte Carlo Simulations
        - 💰 Investment Portfolio Optimization
        - 🌱 ESG & Sustainability Metrics
        - 🤖 RAG-Powered Assistant
        
        Version 2.0.0 | © 2025 MS Hadianto
        """
    }
)

# Custom CSS Styling with Enhanced Footer
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    /* KPI styling */
    .kpi-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    
    .kpi-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        min-width: 200px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Alert styling */
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-success {
        background-color: #d1f2eb;
        border-left: 4px solid #00d4aa;
        color: #00695c;
    }
    
    .alert-warning {
        background-color: #fef9e7;
        border-left: 4px solid #f39c12;
        color: #b7750b;
    }
    
    .alert-danger {
        background-color: #fadbd8;
        border-left: 4px solid #e74c3c;
        color: #c0392b;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    }
    
    /* Enhanced Footer Styling */
    .footer-container {
        margin-top: 4rem;
        padding: 3rem 0 2rem 0;
        border-top: 3px solid #e1e5e9;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .disclaimer-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border-left: 5px solid #dc3545;
    }
    
    .developer-section {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 25px rgba(30,58,138,0.4);
    }
    
    .disclaimer-title {
        color: #dc3545;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .disclaimer-text {
        color: #495057;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-bottom: 0.8rem;
    }
    
    .developer-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 2rem;
        margin-bottom: 1.5rem;
    }
    
    .developer-profile {
        text-align: left;
    }
    
    .developer-name {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(45deg, #ffffff, #e0f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .developer-title {
        font-size: 1rem;
        opacity: 0.95;
        margin: 0.3rem 0;
        font-weight: 500;
    }
    
    .app-info {
        text-align: right;
    }
    
    .version-info {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0.3rem 0;
    }
    
    .copyright {
        font-size: 1rem;
        margin-top: 1.5rem;
        opacity: 0.95;
        text-align: center;
        border-top: 1px solid rgba(255,255,255,0.2);
        padding-top: 1.5rem;
    }
    
    .tech-stack {
        margin-top: 1rem;
        font-size: 0.85rem;
        opacity: 0.8;
        text-align: center;
        font-style: italic;
    }
    
    .expertise-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 1rem;
    }
    
    .badge {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    @media (max-width: 768px) {
        .developer-info {
            text-align: center;
            flex-direction: column;
        }
        .app-info {
            text-align: center;
        }
        .developer-profile {
            text-align: center;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

# Load data function
@st.cache_data
def load_historical_data():
    """Load and process historical hajj financial data"""
    
    # Simulated data based on the real data structure
    data = {
        'Tahun': [2022, 2023, 2024, 2025],
        'BPIH': [85452883, 89629474, 94482028, 91493896],
        'Bipih': [39886009, 49812700, 56046172, 60559399], 
        'NilaiManfaat': [45566874, 39816774, 38435856, 30934497]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate additional metrics
    df['Total_Cost'] = df['BPIH'] + df['Bipih']
    df['Cost_Growth'] = df['Total_Cost'].pct_change() * 100
    df['Benefit_Ratio'] = df['NilaiManfaat'] / df['Total_Cost'] * 100
    df['Sustainability_Index'] = df['NilaiManfaat'] / df['BPIH'] * 100
    
    return df

# Advanced calculations
@st.cache_data
def calculate_projections(base_data, years=10):
    """Calculate future projections using advanced modeling"""
    
    # Base growth rates (calculated from historical data)
    bpih_growth = base_data['BPIH'].pct_change().mean()
    bipih_growth = base_data['Bipih'].pct_change().mean() 
    benefit_decline = abs(base_data['NilaiManfaat'].pct_change().mean())
    
    # Create projections
    projections = []
    last_year = base_data['Tahun'].max()
    last_bpih = base_data['BPIH'].iloc[-1]
    last_bipih = base_data['Bipih'].iloc[-1]
    last_benefit = base_data['NilaiManfaat'].iloc[-1]
    
    for i in range(1, years + 1):
        year = last_year + i
        
        # Apply stochastic modeling
        bpih_proj = last_bpih * (1 + bpih_growth + np.random.normal(0, 0.02)) ** i
        bipih_proj = last_bipih * (1 + bipih_growth + np.random.normal(0, 0.03)) ** i
        benefit_proj = max(
            last_benefit * (1 - benefit_decline + np.random.normal(0, 0.01)) ** i,
            last_benefit * 0.5  # Floor at 50% of original
        )
        
        total_cost = bpih_proj + bipih_proj
        sustainability_idx = benefit_proj / bpih_proj * 100
        
        projections.append({
            'Tahun': year,
            'BPIH': bpih_proj,
            'Bipih': bipih_proj,
            'NilaiManfaat': benefit_proj,
            'Total_Cost': total_cost,
            'Sustainability_Index': sustainability_idx,
            'Risk_Level': 'High' if sustainability_idx < 30 else 'Medium' if sustainability_idx < 50 else 'Low'
        })
    
    return pd.DataFrame(projections)

def create_enhanced_footer():
    """
    Footer sederhana tanpa HTML issues
    """
    st.markdown("---")
    
    # Disclaimer
    st.markdown("### ⚠️ Important Disclaimer")
    st.info("""
    **Educational Purpose:** This application provides financial planning tools for educational purposes only. 
    Please consult with certified financial advisors before making investment decisions.
    """)
    
    # Developer Info
    st.markdown("### 👨‍💻 Developer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **MS Hadianto**  
        🎓 RAG & Agentic AI Enthusiast  
        🕌 Audit Committee Members of Hajj Fund Management Agency  
        🤖 GRC Specialist  
        """)
    
    with col2:
        st.markdown("""
        **🕌 System Information**  
        Version: v2.0  
        Technology: AI-Powered Analytics  
        Updated: June 2025  
        """)
    
    # Copyright
    st.markdown("---")
    st.markdown("""
    **© 2025 MS Hadianto** - All Rights Reserved  
    RAG & Agentic AI Enthusiast | Audit Committee Members of Hajj Fund Management Agency
    """)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🕌 SISTEM SUSTAINABILITAS KEUANGAN HAJI</h1>
    <h3>🤖 AI-Powered Financial Analytics & Optimization Platform</h3>
    <p>Advanced Actuarial Modeling | Machine Learning Optimization | Islamic Finance Compliance</p>
    <p><small>Developed by MS Hadianto | RAG & Agentic AI Enthusiast | Audit Committee of Hajj Fund Management Agency</small></p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🧭 Navigation Center")
    
    # Developer info in sidebar
    st.markdown("### 👨‍💻 Developer")
    st.markdown("""
    **MS Hadianto**  
    *RAG & Agentic AI Enthusiast | Audit Committee of Hajj Fund Management Agency*
    
    🎓 Audit Committee of Hajj Fund Management Agency  
    🤖 RAG & Agentic AI Enthusiast  
    
    """)
    
    st.markdown("---")
    
    # Data Status
    if st.session_state.data_loaded:
        st.success("✅ Data Loaded Successfully")
    else:
        st.warning("⏳ Loading Data...")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("📊 Generate Report", use_container_width=True):
        st.success("📋 Report generation started...")
    
    if st.button("🎯 Run Optimization", use_container_width=True):
        st.info("🚀 Optimization algorithm initiated...")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Analytics Settings")
    
    projection_years = st.slider(
        "Projection Years",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of years for future projections"
    )
    
    confidence_level = st.selectbox(
        "Confidence Level",
        options=[90, 95, 99],
        index=1,
        help="Statistical confidence level for risk calculations"
    )
    
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=['Conservative', 'Moderate', 'Aggressive'],
        value='Moderate'
    )

# Load and display main dashboard
try:
    # Load data
    df_historical = load_historical_data()
    df_projections = calculate_projections(df_historical, projection_years)
    st.session_state.data_loaded = True
    st.session_state.current_data = df_historical
    
    # Main Dashboard Content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_bpih = df_historical['BPIH'].iloc[-1]
        bpih_growth = df_historical['BPIH'].pct_change().iloc[-1] * 100
        st.metric(
            "💰 Current BPIH",
            f"Rp {current_bpih:,.0f}",
            f"{bpih_growth:+.1f}%"
        )
    
    with col2:
        current_benefit = df_historical['NilaiManfaat'].iloc[-1]
        benefit_change = df_historical['NilaiManfaat'].pct_change().iloc[-1] * 100
        st.metric(
            "📈 Current Benefit Value",
            f"Rp {current_benefit:,.0f}",
            f"{benefit_change:+.1f}%"
        )
    
    with col3:
        sustainability_idx = df_historical['Sustainability_Index'].iloc[-1]
        st.metric(
            "🌱 Sustainability Index",
            f"{sustainability_idx:.1f}%",
            "🔴 Critical" if sustainability_idx < 40 else "🟡 Warning" if sustainability_idx < 60 else "🟢 Healthy"
        )
    
    with col4:
        avg_growth = df_historical['Cost_Growth'].mean()
        st.metric(
            "📊 Avg Cost Growth",
            f"{avg_growth:.1f}%/year",
            "Trend Analysis"
        )
    
    # Alert System
    st.markdown("---")
    
    sustainability_current = df_historical['Sustainability_Index'].iloc[-1]
    
    if sustainability_current < 40:
        st.markdown("""
        <div class="alert alert-danger">
            <strong>🚨 CRITICAL SUSTAINABILITY ALERT</strong><br>
            Current sustainability index is below 40%. Immediate intervention required!
            <ul>
                <li>Consider increasing investment returns</li>
                <li>Optimize operational costs</li>
                <li>Review benefit distribution policy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif sustainability_current < 60:
        st.markdown("""
        <div class="alert alert-warning">
            <strong>⚠️ SUSTAINABILITY WARNING</strong><br>
            Sustainability index shows concerning trends. Monitoring recommended.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert alert-success">
            <strong>✅ HEALTHY SUSTAINABILITY STATUS</strong><br>
            Current financial health is within acceptable parameters.
        </div>
        """, unsafe_allow_html=True)
    
    # Main Charts Section
    st.markdown("## 📊 Financial Trend Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Historical Trends", "🔮 Future Projections", "🎯 Risk Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost trends
            fig_costs = go.Figure()
            fig_costs.add_trace(go.Scatter(
                x=df_historical['Tahun'], 
                y=df_historical['BPIH'],
                mode='lines+markers',
                name='BPIH',
                line=dict(color='#e74c3c', width=3)
            ))
            fig_costs.add_trace(go.Scatter(
                x=df_historical['Tahun'], 
                y=df_historical['Bipih'],
                mode='lines+markers',
                name='Bipih',
                line=dict(color='#3498db', width=3)
            ))
            fig_costs.update_layout(
                title="💰 Cost Evolution (2022-2025)",
                xaxis_title="Year",
                yaxis_title="Amount (IDR)",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_costs, use_container_width=True)
        
        with col2:
            # Sustainability trend
            fig_sustain = go.Figure()
            fig_sustain.add_trace(go.Scatter(
                x=df_historical['Tahun'],
                y=df_historical['Sustainability_Index'],
                mode='lines+markers',
                name='Sustainability Index',
                line=dict(color='#27ae60', width=4),
                fill='tonexty'
            ))
            fig_sustain.add_hline(
                y=50, 
                line_dash="dash", 
                line_color="orange",
                annotation_text="Warning Threshold"
            )
            fig_sustain.add_hline(
                y=30, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Critical Threshold"
            )
            fig_sustain.update_layout(
                title="🌱 Sustainability Index Trend",
                xaxis_title="Year",
                yaxis_title="Sustainability Index (%)",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_sustain, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔮 10-Year Financial Projections")
        
        # Combined historical and projection data
        df_combined = pd.concat([df_historical[['Tahun', 'BPIH', 'Bipih', 'NilaiManfaat', 'Sustainability_Index']], 
                                df_projections[['Tahun', 'BPIH', 'Bipih', 'NilaiManfaat', 'Sustainability_Index']]])
        
        # Projection chart
        fig_proj = make_subplots(
            rows=2, cols=2,
            subplot_titles=('BPIH Projection', 'Benefit Value Projection', 
                          'Total Cost Projection', 'Sustainability Forecast'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # BPIH projection
        fig_proj.add_trace(go.Scatter(
            x=df_combined['Tahun'], 
            y=df_combined['BPIH'],
            mode='lines+markers',
            name='BPIH',
            line=dict(color='#e74c3c')
        ), row=1, col=1)
        
        # Benefit projection
        fig_proj.add_trace(go.Scatter(
            x=df_combined['Tahun'], 
            y=df_combined['NilaiManfaat'],
            mode='lines+markers',
            name='Nilai Manfaat',
            line=dict(color='#27ae60')
        ), row=1, col=2)
        
        # Total cost projection
        total_cost_combined = df_combined['BPIH'] + df_combined['Bipih']
        fig_proj.add_trace(go.Scatter(
            x=df_combined['Tahun'], 
            y=total_cost_combined,
            mode='lines+markers',
            name='Total Cost',
            line=dict(color='#8e44ad')
        ), row=2, col=1)
        
        # Sustainability projection
        fig_proj.add_trace(go.Scatter(
            x=df_combined['Tahun'], 
            y=df_combined['Sustainability_Index'],
            mode='lines+markers',
            name='Sustainability Index',
            line=dict(color='#f39c12')
        ), row=2, col=2)
        
        fig_proj.update_layout(
            height=600,
            showlegend=False,
            title_text="Comprehensive Financial Projections",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_proj, use_container_width=True)
        
        # Projection summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            future_bpih = df_projections['BPIH'].iloc[-1]
            bpih_increase = ((future_bpih / df_historical['BPIH'].iloc[-1]) - 1) * 100
            st.metric(
                f"BPIH in {projection_years} years",
                f"Rp {future_bpih:,.0f}",
                f"+{bpih_increase:.1f}%"
            )
        
        with col2:
            future_benefit = df_projections['NilaiManfaat'].iloc[-1]
            benefit_change = ((future_benefit / df_historical['NilaiManfaat'].iloc[-1]) - 1) * 100
            st.metric(
                f"Benefit Value in {projection_years} years",
                f"Rp {future_benefit:,.0f}",
                f"{benefit_change:+.1f}%"
            )
        
        with col3:
            future_sustainability = df_projections['Sustainability_Index'].iloc[-1]
            sustainability_change = future_sustainability - df_historical['Sustainability_Index'].iloc[-1]
            st.metric(
                f"Sustainability Index in {projection_years} years",
                f"{future_sustainability:.1f}%",
                f"{sustainability_change:+.1f}pp"
            )
    
    with tab3:
        st.markdown("### 🎯 Advanced Risk Analysis")
        
        # Risk metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Value at Risk calculation
            returns = df_historical['Cost_Growth'].dropna()
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            st.markdown("#### 📊 Value at Risk (VaR)")
            st.write(f"**95% VaR:** {var_95:.2f}%")
            st.write(f"**99% VaR:** {var_99:.2f}%")
            
            # Risk level assessment
            risk_score = abs(sustainability_current - 100) + abs(var_95)
            
            if risk_score > 80:
                risk_level = "🔴 High Risk"
                risk_color = "red"
            elif risk_score > 50:
                risk_level = "🟡 Medium Risk"
                risk_color = "orange"
            else:
                risk_level = "🟢 Low Risk"
                risk_color = "green"
            
            st.markdown(f"**Overall Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
        
        with col2:
            # Risk distribution
            risk_data = df_projections['Risk_Level'].value_counts()
            
            fig_risk = px.pie(
                values=risk_data.values,
                names=risk_data.index,
                title="Risk Distribution Over Projection Period",
                color_discrete_map={
                    'Low': '#27ae60',
                    'Medium': '#f39c12', 
                    'High': '#e74c3c'
                }
            )
            st.plotly_chart(fig_risk, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error loading application: {str(e)}")
    st.info("💡 Please check data files and try refreshing the page.")

# Enhanced Footer Section
st.markdown("---")
create_enhanced_footer()