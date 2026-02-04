# ============================================================================
# PAGES/02_🔮_Projections.py - Financial Projections Page (REVISED)
# ============================================================================

"""
🔮 FINANCIAL PROJECTIONS
Advanced forecasting and scenario modeling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🔮 Projections",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .projections-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .scenario-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #3498db;
        height: 100%; /* --- Tambahan untuk menyamakan tinggi card --- */
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
    
    .model-performance {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="projections-header">
    <h1>🔮 FINANCIAL PROJECTIONS & FORECASTING</h1>
    <h3>Advanced Modeling for Strategic Planning</h3>
    <p>Scenario Analysis | Trend Forecasting | Predictive Modeling | Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("## 🔮 Projection Controls")
    
    projection_model = st.selectbox(
        "Forecasting Model",
        [
            "Linear Trend",
            "Polynomial Trend", 
            "Exponential Smoothing",
        ]
    )
    
    projection_years = st.slider(
        "Projection Years",
        min_value=5,
        max_value=25,
        value=10
    )
    
    confidence_level = st.selectbox(
        "Confidence Level",
        [90, 95, 99],
        index=1
    )
    
# Data loading and preparation
@st.cache_data
def load_projection_data():
    """Load historical data for projections"""
    np.random.seed(42)
    
    years = list(range(2015, 2026))
    data = []
    
    base_bpih = 70000000
    base_bipih = 32000000
    base_benefit = 48000000
    
    for i, year in enumerate(years):
        cycle = np.sin(i * 0.4) * 0.015
        trend = 0.05
        noise = np.random.normal(0, 0.02)
        growth_rate = trend + cycle + noise
        
        bpih = base_bpih * (1 + growth_rate) ** i
        bipih = base_bipih * (1 + growth_rate + 0.01) ** i
        benefit = base_benefit * (1 + growth_rate - 0.015) ** i
        
        data.append({
            'Year': year,
            'BPIH': bpih,
            'Bipih': bipih,
            'NilaiManfaat': benefit,
            'Total_Cost': bpih + bipih,
            'Sustainability_Index': (benefit / bpih) * 100,
            'Growth_Rate': growth_rate
        })
    
    return pd.DataFrame(data)

class FinancialProjector:
    """Financial projection engine"""
    
    def __init__(self, data):
        self.data = data
        self.models = {}
    
    def linear_trend_projection(self, column, years, conf_level):
        X = np.array(self.data.index).reshape(-1, 1)
        y = self.data[column].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.array(range(len(self.data), len(self.data) + years)).reshape(-1, 1)
        projections = model.predict(future_X)
        
        residuals = y - model.predict(X)
        std_error = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
        
        z_score = stats.norm.ppf(1 - (1 - conf_level/100) / 2)
        margin_error = z_score * std_error
        
        return {
            'projections': projections,
            'lower_bound': projections - margin_error,
            'upper_bound': projections + margin_error,
            'r_squared': model.score(X, y),
            'model': model
        }
    
    def polynomial_trend_projection(self, column, years, degree=2, conf_level=95):
        X = np.array(self.data.index).reshape(-1, 1)
        y = self.data[column].values

        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        future_X = np.array(range(len(self.data), len(self.data) + years)).reshape(-1, 1)
        future_X_poly = poly_features.transform(future_X)
        projections = model.predict(future_X_poly)

        # Calculate confidence intervals
        residuals = y - model.predict(X_poly)
        std_error = np.sqrt(np.sum(residuals**2) / (len(y) - degree - 1))
        z_score = stats.norm.ppf(1 - (1 - conf_level/100) / 2)
        margin_error = z_score * std_error

        return {
            'projections': projections,
            'lower_bound': projections - margin_error,
            'upper_bound': projections + margin_error,
            'r_squared': model.score(X_poly, y),
            'model': model,
            'poly_features': poly_features
        }
    
    def exponential_smoothing_projection(self, column, years, alpha=0.3):
        values = self.data[column].values
        smoothed = [values[0]]
        
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
        
        last_smoothed = smoothed[-1]
        trend = np.mean(np.diff(smoothed[-5:]))
        
        projections = [last_smoothed + trend * (i + 1) for i in range(years)]
        
        return {
            'projections': np.array(projections),
            'smoothed_historical': smoothed,
            'trend': trend
        }

# Load data and initialize projector
data = load_projection_data()
projector = FinancialProjector(data)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend Analysis", 
    "🎯 Scenario Modeling", 
    "🔬 Model Comparison",
    "📋 Strategic Insights"
])

with tab1:
    st.markdown("## 📈 Trend Analysis & Base Projections")
    
    metric_to_project = st.selectbox(
        "Select Metric for Projection",
        ['BPIH', 'Bipih', 'NilaiManfaat', 'Total_Cost', 'Sustainability_Index']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if projection_model == "Linear Trend":
            results = projector.linear_trend_projection(metric_to_project, projection_years, confidence_level)
            st.markdown(f"""
            <div class="model-performance">
                <h4>📊 Linear Trend Model Performance</h4>
                <p><strong>R-squared:</strong> {results.get('r_squared', 0):.3f}</p>
                <p><strong>Model Type:</strong> Linear Regression</p>
                <p><strong>Confidence Level:</strong> {confidence_level}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif projection_model == "Polynomial Trend":
            results = projector.polynomial_trend_projection(metric_to_project, projection_years, conf_level=confidence_level)
            st.markdown(f"""
            <div class="model-performance">
                <h4>📊 Polynomial Trend Model Performance</h4>
                <p><strong>R-squared:</strong> {results.get('r_squared', 0):.3f}</p>
                <p><strong>Model Type:</strong> Polynomial (degree 2)</p>
                <p><strong>Confidence Level:</strong> {confidence_level}%</p>
                <p><strong>Flexibility:</strong> Higher curve fitting</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif projection_model == "Exponential Smoothing":
            results = projector.exponential_smoothing_projection(metric_to_project, projection_years)
            st.markdown(f"""
            <div class="model-performance">
                <h4>📊 Exponential Smoothing Performance</h4>
                <p><strong>Trend:</strong> {results.get('trend', 0):,.0f} per year</p>
                <p><strong>Model Type:</strong> Exponential Smoothing</p>
                <p><strong>Alpha:</strong> 0.3 (smoothing parameter)</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'projections' in results:
            current_value = data[metric_to_project].iloc[-1]
            final_projection = results['projections'][-1]
            total_change = ((final_projection - current_value) / current_value) * 100 if current_value != 0 else 0
            annual_growth = (total_change / projection_years)
            
            st.markdown(f"""
            <div class="projection-result">
                <h3>🔮 {projection_years}-Year Projection</h3>
                <p><strong>Current Value:</strong> {current_value:,.0f}</p>
                <p><strong>Projected Value:</strong> {final_projection:,.0f}</p>
                <p><strong>Total Change:</strong> {total_change:+.1f}%</p>
                <p><strong>Annual Growth:</strong> {annual_growth:+.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Projection Visualization")
    
    if 'projections' in results:
        historical_years = data['Year'].tolist()
        future_years = list(range(data['Year'].max() + 1, data['Year'].max() + projection_years + 1))
        
        fig_projection = go.Figure()
        
        fig_projection.add_trace(go.Scatter(
            x=historical_years, y=data[metric_to_project],
            mode='lines+markers', name='Historical Data',
            line=dict(color='#3498db', width=3)
        ))
        
        fig_projection.add_trace(go.Scatter(
            x=future_years, y=results['projections'],
            mode='lines+markers', name=f'{projection_model} Projection',
            line=dict(color='#e74c3c', width=3, dash='dash')
        ))
        
        if 'lower_bound' in results and 'upper_bound' in results:
            fig_projection.add_trace(go.Scatter(
                x=future_years + future_years[::-1],
                y=np.concatenate([results['upper_bound'], results['lower_bound'][::-1]]),
                fill='toself', fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True, name=f'{confidence_level}% Confidence Interval'
            ))
        
        fig_projection.update_layout(
            title=f"{metric_to_project} Projection ({projection_model})",
            xaxis_title="Year", yaxis_title="Value",
            template="plotly_white", height=500
        )
        st.plotly_chart(fig_projection, use_container_width=True)

with tab2:
    st.markdown("## 🎯 Scenario Modeling & Analysis")

    # Custom Scenario Builder
    with st.expander("🔧 Create Custom Scenario", expanded=False):
        st.markdown("**Build Your Own Scenario:**")
        col_custom1, col_custom2 = st.columns(2)

        with col_custom1:
            custom_name = st.text_input("Scenario Name", "My Custom Scenario", key="custom_scenario_name")
            custom_bpih_adj = st.slider("BPIH Adjustment (%)", -20, 30, 0, key="custom_bpih") / 100 + 1
            custom_bipih_adj = st.slider("Bipih Adjustment (%)", -20, 30, 0, key="custom_bipih") / 100 + 1

        with col_custom2:
            custom_description = st.text_input("Description", "User-defined scenario", key="custom_desc")
            custom_benefit_adj = st.slider("Benefit Adjustment (%)", -30, 30, 0, key="custom_benefit") / 100 + 1

        include_custom = st.checkbox("Include Custom Scenario in Analysis", value=False, key="include_custom")

    # Base scenario parameters
    scenario_params = {
        'pessimistic': {
            'name': 'Pessimistic Scenario',
            'description': 'Higher costs, lower returns.',
            'adjustments': {'BPIH': 1.15, 'Bipih': 1.12, 'NilaiManfaat': 0.90, 'color': '#e74c3c'}
        },
        'baseline': {
            'name': 'Baseline Scenario',
            'description': 'Current trend continuation.',
            'adjustments': {'BPIH': 1.0, 'Bipih': 1.0, 'NilaiManfaat': 1.0, 'color': '#3498db'}
        },
        'optimistic': {
            'name': 'Optimistic Scenario',
            'description': 'Cost efficiency, higher returns.',
            'adjustments': {'BPIH': 0.95, 'Bipih': 0.93, 'NilaiManfaat': 1.15, 'color': '#27ae60'}
        }
    }

    # Add custom scenario if selected
    if include_custom:
        scenario_params['custom'] = {
            'name': custom_name,
            'description': custom_description,
            'adjustments': {'BPIH': custom_bpih_adj, 'Bipih': custom_bipih_adj, 'NilaiManfaat': custom_benefit_adj, 'color': '#9b59b6'}
        }
    
    scenario_projections = {}
    
    for scenario_name, params in scenario_params.items():
        bpih_proj = projector.linear_trend_projection('BPIH', projection_years, 95)['projections']
        benefit_proj = projector.linear_trend_projection('NilaiManfaat', projection_years, 95)['projections']
        
        # --- PERBAIKAN: Menggunakan .get() untuk mengakses adjustments dengan aman ---
        adj = params.get('adjustments', {})
        adjusted_bpih = bpih_proj * adj.get('BPIH', 1.0)
        adjusted_benefit = benefit_proj * adj.get('NilaiManfaat', 1.0)
        
        adjusted_sustainability = (adjusted_benefit / adjusted_bpih) * 100 if np.all(adjusted_bpih != 0) else np.zeros_like(adjusted_bpih)
        
        scenario_projections[scenario_name] = {
            'sustainability': adjusted_sustainability,
            'params': params
        }
    
    st.markdown("### 📊 Scenario Comparison")
    fig_scenarios = go.Figure()
    future_years = list(range(data['Year'].max() + 1, data['Year'].max() + projection_years + 1))
    
    for scenario_name, scenario_data in scenario_projections.items():
        params = scenario_data.get('params', {})
        # --- PERBAIKAN: Mengakses 'color' dari dalam 'adjustments' ---
        color = params.get('adjustments', {}).get('color', '#3498db')
        
        fig_scenarios.add_trace(go.Scatter(
            x=future_years, y=scenario_data['sustainability'],
            mode='lines+markers', name=params.get('name', 'Unnamed'),
            line=dict(color=color, width=3)
        ))
    
    fig_scenarios.add_hline(y=70, line_dash="dot", line_color="green", annotation_text="Excellent (70%)")
    fig_scenarios.add_hline(y=50, line_dash="dot", line_color="orange", annotation_text="Good (50%)")
    fig_scenarios.add_hline(y=30, line_dash="dot", line_color="red", annotation_text="Critical (30%)")
    
    fig_scenarios.update_layout(
        title="Sustainability Index - Scenario Analysis",
        xaxis_title="Year", yaxis_title="Sustainability Index (%)",
        template="plotly_white", height=500
    )
    st.plotly_chart(fig_scenarios, use_container_width=True)
    
    st.markdown("### 📋 Scenario Summary")
    cols = st.columns(len(scenario_projections))
    
    for i, (scenario_name, scenario_data) in enumerate(scenario_projections.items()):
        with cols[i]:
            params = scenario_data.get('params', {})
            final_sustainability = scenario_data['sustainability'][-1]
            
            # --- PERBAIKAN: Mengakses semua data params dengan .get() untuk keamanan ---
            adj = params.get('adjustments', {})
            color = adj.get('color', 'black')
            name = params.get('name', 'Unnamed Scenario')
            description = params.get('description', '-')
            bpih_adj = (adj.get('BPIH', 1.0) - 1) * 100
            benefit_adj = (adj.get('NilaiManfaat', 1.0) - 1) * 100
            
            st.markdown(f"""
            <div class="scenario-card">
                <h4 style="color: {color};">{name}</h4>
                <p><strong>Final Sustainability:</strong> {final_sustainability:.1f}%</p>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Key Assumptions:</strong></p>
                <ul>
                    <li>BPIH: {bpih_adj:+.0f}%</li>
                    <li>Benefit: {benefit_adj:+.0f}%</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("## 🔬 Model Comparison & Validation")
    
    models_to_compare = ['Linear Trend', 'Polynomial Trend', 'Exponential Smoothing']
    comparison_results = {}
    
    for model in models_to_compare:
        if model == 'Linear Trend':
            result = projector.linear_trend_projection('Sustainability_Index', projection_years, 95)
        elif model == 'Polynomial Trend':
            result = projector.polynomial_trend_projection('Sustainability_Index', projection_years)
        elif model == 'Exponential Smoothing':
            result = projector.exponential_smoothing_projection('Sustainability_Index', projection_years)
        
        comparison_results[model] = result
    
    fig_comparison = go.Figure()
    future_years = list(range(data['Year'].max() + 1, data['Year'].max() + projection_years + 1))
    colors = ['#e74c3c', '#27ae60', '#f39c12']
    
    for i, (model_name, results) in enumerate(comparison_results.items()):
        if 'projections' in results:
            fig_comparison.add_trace(go.Scatter(
                x=future_years, y=results['projections'],
                mode='lines+markers', name=model_name,
                line=dict(color=colors[i], width=3)
            ))
    
    fig_comparison.update_layout(
        title="Model Comparison - Sustainability Index Projections",
        xaxis_title="Year", yaxis_title="Sustainability Index (%)",
        template="plotly_white", height=500
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("### 📊 Model Performance Metrics")
    performance_data = []
    
    for model_name, results in comparison_results.items():
        r_squared = results.get('r_squared')
        final_value = results.get('projections', [None])[-1]
        performance_data.append({
            'Model': model_name,
            'R-squared': f"{r_squared:.3f}" if r_squared is not None else 'N/A',
            'Suitability': 'High' if r_squared and r_squared > 0.8 else 'Medium' if r_squared and r_squared > 0.6 else 'Low' if r_squared else 'N/A',
            'Final Value': f"{final_value:.1f}%" if final_value is not None else 'N/A'
        })
    
    st.dataframe(pd.DataFrame(performance_data), use_container_width=True)

with tab4:
    st.markdown("## 📋 Strategic Insights & Recommendations")
    
    baseline_sustainability = scenario_projections.get('baseline', {}).get('sustainability', [0])
    final_baseline = baseline_sustainability[-1]
    
    st.markdown("### ⚠️ Risk Assessment")
    
    if final_baseline < 40:
        insight = {'level': 'Critical', 'message': f'Projected sustainability index ({final_baseline:.1f}%) indicates critical risk', 'recommendations': ['Immediate strategic intervention required', 'Consider emergency cost reduction measures', 'Review and optimize investment strategy']}
    elif final_baseline < 60:
        insight = {'level': 'Warning', 'message': f'Projected sustainability index ({final_baseline:.1f}%) shows concerning trend', 'recommendations': ['Monitor trends closely', 'Implement gradual optimization measures', 'Review cost structure and efficiency']}
    else:
        insight = {'level': 'Healthy', 'message': f'Projected sustainability index ({final_baseline:.1f}%) within acceptable range', 'recommendations': ['Continue current strategy', 'Monitor for optimization opportunities', 'Maintain balanced approach']}
    
    level_colors = {'Critical': '#e74c3c', 'Warning': '#f39c12', 'Healthy': '#27ae60'}
    # --- PERBAIKAN: Menggunakan .get() untuk keamanan ---
    color = level_colors.get(insight['level'], 'black')
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {color}22 0%, {color}11 100%); border-left: 4px solid {color}; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="color: {color}; margin-top: 0;">{insight['level']} Assessment</h4>
        <p><strong>{insight['message']}</strong></p>
        <h5>Recommended Actions:</h5>
        <ul>{''.join([f'<li>{rec}</li>' for rec in insight.get("recommendations", [])])}</ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Strategic Planning Matrix")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📈 Opportunities")
        st.markdown("""
        - Optimize investment portfolio allocation
        - Implement cost efficiency programs
        - Enhance digital transformation
        - Develop alternative revenue streams
        """)
    with col2:
        st.markdown("#### ⚠️ Challenges") 
        st.markdown("""
        - Rising operational costs
        - Economic uncertainty and volatility
        - Regulatory compliance requirements
        - Technology adoption and implementation
        """)
    
    st.markdown("### 📅 Implementation Timeline")
    timeline_data = [
        {"Phase": "Immediate (0-6 months)", "Actions": "Cost optimization, risk assessment"},
        {"Phase": "Short-term (6-18 months)", "Actions": "Strategy implementation, system upgrades"},
        {"Phase": "Medium-term (1-3 years)", "Actions": "Portfolio restructuring, technology integration"},
        {"Phase": "Long-term (3-5 years)", "Actions": "Strategic transformation, sustainability assurance"}
    ]
    st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <h4>🔮 Financial Projections & Strategic Planning</h4>
    <p>Advanced forecasting models for informed decision making</p>
    <p><em>Data-driven insights for sustainable hajj financing</em></p>
</div>
""", unsafe_allow_html=True)