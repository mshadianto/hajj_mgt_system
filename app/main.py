"""
🕌 APLIKASI SUSTAINABILITAS KEUANGAN HAJI
Main Streamlit Application with Advanced Features + Clean Landing Page

Author: MS Hadianto - RAG & Agentic AI Enthusiast | Audit Committee of Hajj Fund Management Agency
Version: 2.1.1 - Fixed HTML Rendering Issues
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
    initial_sidebar_state="collapsed" if not st.session_state.get('show_dashboard', False) else "expanded",
    menu_items={
        'Get Help': 'https://hajj-finance-support.com',
        'Report a bug': 'https://hajj-finance-support.com/bug',
        'About': """
        # Hajj Financial Sustainability Application
        
        **Advanced AI-Powered Financial Planning for Hajj**
        
        Developed by: MS Hadianto
        RAG & Agentic AI Enthusiast | Audit Committee of Hajj Fund Management Agency
        
        Version 2.1.1 | © 2025 MS Hadianto
        """
    }
)

# Initialize session state for landing page
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

# Clean CSS Styling - Simplified
st.markdown("""
<style>
    /* Landing Page Gradient Header */
    .landing-header {
        background: linear-gradient(135deg, #2E3192 0%, #1BFFFF 100%);
        padding: 3rem 2rem;
        text-align: center;
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .landing-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .landing-subtitle {
        font-size: 1.3rem;
        margin-bottom: 1rem;
        opacity: 0.95;
    }
    
    .landing-description {
        font-size: 1rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .landing-author {
        font-size: 0.9rem;
        opacity: 0.8;
        font-style: italic;
    }
    
    /* Feature Cards */
    .feature-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border-color: #3B82F6;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    
    .feature-description {
        color: #6c757d;
        line-height: 1.6;
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
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #2E3192 0%, #1BFFFF 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46, 49, 146, 0.3);
    }
    
    /* Main dashboard header */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide Streamlit branding */
    .css-1rs6os.edgvbvh3, .css-10trblm.e16nr0p30 {
        display: none;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .landing-title {
            font-size: 2rem;
        }
        .landing-subtitle {
            font-size: 1.1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def show_landing_page():
    """Clean Landing Page using Native Streamlit Components"""
    
    # Landing Header
    st.markdown("""
    <div class="landing-header">
        <h1 class="landing-title">🕌 SISTEM SUSTAINABILITAS KEUANGAN HAJI</h1>
        <p class="landing-subtitle">🚀 AI-Powered Financial Analytics & Optimization Platform</p>
        <p class="landing-description">Advanced Actuarial Modeling | Machine Learning Optimization | Islamic Finance Compliance</p>
        <p class="landing-author">Developed by MS Hadianto | RAG & Agentic AI Enthusiast | Audit Committee of Hajj Fund Management Agency</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick KPI Overview using native Streamlit metrics
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🌱 Sustainability Index", 
            value="47.9%", 
            delta="0.4%",
            delta_color="normal"
        )
        st.caption("⚠️ Warning Status")
    
    with col2:
        st.metric(
            label="💰 Total Cost", 
            value="Rp 0.2B", 
            delta="4.2%",
            delta_color="inverse"
        )
        st.caption("📈 Rising Trend")
    
    with col3:
        st.metric(
            label="📊 Investment Return", 
            value="1.7%", 
            delta="-5.3%",
            delta_color="inverse"
        )
        st.caption("🎯 Target: 7.0%")
    
    with col4:
        st.metric(
            label="⚠️ Risk Level", 
            value="3", 
            delta="Stable"
        )
        st.caption("🟢 Low Risk")
    
    st.markdown("---")
    
    # Features Section using native Streamlit
    st.markdown("### ✨ Platform Capabilities")
    st.markdown("*Comprehensive financial analytics dan AI-powered optimization untuk sustainable hajj fund management*")
    
    # Create feature cards using columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Executive Dashboard</h3>
            <p class="feature-description">Real-time monitoring KPI finansial dengan advanced analytics dan performance tracking</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3 class="feature-title">AI Optimization</h3>
            <p class="feature-description">Genetic algorithm optimization untuk portfolio management dan risk mitigation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔮</div>
            <h3 class="feature-title">Financial Projections</h3>
            <p class="feature-description">Advanced forecasting models dengan proyeksi hingga 25 tahun untuk strategic planning</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h3 class="feature-title">Advanced Analytics</h3>
            <p class="feature-description">Statistical modeling dan machine learning insights untuk informed decision making</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3 class="feature-title">RAG Assistant</h3>
            <p class="feature-description">AI-powered assistant dengan retrieval-augmented generation untuk financial guidance</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌱</div>
            <h3 class="feature-title">ESG Sustainability</h3>
            <p class="feature-description">Comprehensive ESG assessment dengan scoring Environmental, Social, dan Governance</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # CTA Section
    st.markdown("### 🚀 Ready to Optimize Hajj Fund Management?")
    st.markdown("*Experience the power of AI-driven financial analytics dan sustainability monitoring*")
    
    # Enter Dashboard Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 ENTER DASHBOARD", key="enter_dashboard", help="Access the full financial analytics platform"):
            st.session_state.show_dashboard = True
            st.rerun()
    
    # Simple footer
    st.markdown("---")
    st.markdown("**© 2025 MS Hadianto** - RAG & Agentic AI Enthusiast | Audit Committee of Hajj Fund Management Agency")

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
    """Footer sederhana tanpa HTML issues"""
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
        Version: v2.1.1  
        Technology: AI-Powered Analytics  
        Updated: June 2025  
        """)
    
    # Copyright
    st.markdown("---")
    st.markdown("""
    **© 2025 MS Hadianto** - All Rights Reserved  
    RAG & Agentic AI Enthusiast | Audit Committee Members of Hajj Fund Management Agency
    """)

def show_main_dashboard():
    """Main Dashboard Application"""
    
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
        
        # Back to Landing Page Button
        if st.button("🏠 Back to Landing Page", use_container_width=True):
            st.session_state.show_dashboard = False
            st.rerun()
        
        st.markdown("---")
        
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
            st.error("""
            🚨 **CRITICAL SUSTAINABILITY ALERT**  
            Current sustainability index is below 40%. Immediate intervention required!
            
            **Recommendations:**
            - Consider increasing investment returns
            - Optimize operational costs  
            - Review benefit distribution policy
            """)
        elif sustainability_current < 60:
            st.warning("""
            ⚠️ **SUSTAINABILITY WARNING**  
            Sustainability index shows concerning trends. Monitoring recommended.
            """)
        else:
            st.success("""
            ✅ **HEALTHY SUSTAINABILITY STATUS**  
            Current financial health is within acceptable parameters.
            """)
        
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
                if len(returns) > 0:
                    var_95 = np.percentile(returns, 5)
                    var_99 = np.percentile(returns, 1)
                else:
                    var_95 = 0
                    var_99 = 0
                
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
                
                if len(risk_data) > 0:
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
                else:
                    st.info("No risk data available for visualization")

    except Exception as e:
        st.error(f"❌ Error loading application: {str(e)}")
        st.info("💡 Please check data files and try refreshing the page.")

    # Enhanced Footer Section
    st.markdown("---")
    create_enhanced_footer()

# ================================================================
# MAIN APPLICATION LOGIC
# ================================================================

# Check if user wants to see dashboard or landing page
if not st.session_state.show_dashboard:
    # Show Landing Page
    show_landing_page()
else:
    # Show Main Dashboard
    show_main_dashboard()