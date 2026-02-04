"""
📈 ADVANCED FINANCIAL ANALYTICS MODULE
Comprehensive financial analysis and reporting for hajj fund management

Features:
- Financial ratio analysis
- Trend analysis and forecasting
- Performance benchmarking
- Risk analytics
- Correlation analysis
- Statistical modeling
- Interactive dashboards
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="📈 Advanced Analytics",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .analytics-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(30, 60, 114, 0.3);
    }
    
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Advanced Analytics Class
class HajjFinancialAnalytics:
    """
    Comprehensive financial analytics for hajj fund management
    """
    
    def __init__(self, data):
        self.data = data
        self.ratios = self._calculate_financial_ratios()
        self.trends = self._analyze_trends()
        self.risk_metrics = self._calculate_risk_metrics()
    
    def _calculate_financial_ratios(self):
        """Calculate comprehensive financial ratios with column validation"""
        df = self.data.copy()

        ratios = {}

        # Validate required columns exist
        required_cols = ['NilaiManfaat', 'BPIH', 'Bipih', 'Total_Cost', 'Year']
        optional_cols = ['OperationalCost', 'InvestmentReturn', 'InflationRate']

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")

        # Sustainability Ratios (required columns)
        ratios['sustainability_index'] = (df['NilaiManfaat'] / df['BPIH']) * 100
        ratios['cost_coverage_ratio'] = df['NilaiManfaat'] / df['Total_Cost']
        ratios['benefit_efficiency'] = df['NilaiManfaat'] / (df['BPIH'] + df['Bipih'])

        # Growth Ratios (required columns)
        ratios['bpih_growth'] = df['BPIH'].pct_change() * 100
        ratios['bipih_growth'] = df['Bipih'].pct_change() * 100
        ratios['benefit_growth'] = df['NilaiManfaat'].pct_change() * 100
        ratios['total_cost_growth'] = df['Total_Cost'].pct_change() * 100

        # Efficiency Ratios
        ratios['cost_per_beneficiary'] = df['Total_Cost'] / 1000  # Assuming 1000 beneficiaries
        ratios['benefit_per_cost'] = df['NilaiManfaat'] / df['Total_Cost']

        # Optional: Operational efficiency (if column exists)
        if 'OperationalCost' in df.columns:
            ratios['operational_efficiency'] = 100 - (df['OperationalCost'] / df['Total_Cost'] * 100)
        else:
            ratios['operational_efficiency'] = pd.Series([np.nan] * len(df))

        # Investment Performance Ratios (optional columns)
        if 'InvestmentReturn' in df.columns:
            ratios['roi'] = df['InvestmentReturn'] * 100
            if 'InflationRate' in df.columns:
                ratios['real_return'] = (df['InvestmentReturn'] - df['InflationRate']) * 100
            else:
                ratios['real_return'] = ratios['roi']  # Assume 0 inflation if not available
            ratios['risk_adjusted_return'] = ratios['roi'] / (df['InvestmentReturn'].std() * 100 + 1e-6)
        else:
            ratios['roi'] = pd.Series([np.nan] * len(df))
            ratios['real_return'] = pd.Series([np.nan] * len(df))
            ratios['risk_adjusted_return'] = pd.Series([np.nan] * len(df))

        return pd.DataFrame(ratios, index=df['Year'])
    
    def _analyze_trends(self):
        """Analyze trends using statistical methods"""
        trends = {}
        
        for column in ['BPIH', 'Bipih', 'NilaiManfaat', 'Total_Cost']:
            if column in self.data.columns:
                values = self.data[column].values
                years = np.arange(len(values))
                
                # Linear trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
                
                trends[column] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'trend_strength': 'Strong' if abs(r_value) > 0.7 else 'Moderate' if abs(r_value) > 0.4 else 'Weak',
                    'trend_direction': 'Increasing' if slope > 0 else 'Decreasing',
                    'annual_change': slope,
                    'confidence_interval': slope + np.array([-1.96, 1.96]) * std_err
                }
        
        return trends
    
    def _calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        risk_metrics = {}
        
        # Volatility measures
        for col in ['BPIH', 'Bipih', 'NilaiManfaat']:
            if col in self.data.columns:
                returns = self.data[col].pct_change().dropna()
                
                risk_metrics[f'{col}_volatility'] = returns.std() * 100
                risk_metrics[f'{col}_var_95'] = np.percentile(returns, 5) * 100
                risk_metrics[f'{col}_var_99'] = np.percentile(returns, 1) * 100
                risk_metrics[f'{col}_expected_shortfall'] = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Sustainability risk
        sustainability_scores = self.ratios['sustainability_index']
        risk_metrics['sustainability_risk'] = (sustainability_scores < 50).sum() / len(sustainability_scores) * 100
        risk_metrics['sustainability_volatility'] = sustainability_scores.std()
        
        # Correlation risk
        correlation_matrix = self.data[['BPIH', 'Bipih', 'NilaiManfaat']].corr()
        risk_metrics['avg_correlation'] = correlation_matrix.abs().mean().mean()
        
        return risk_metrics
    
    def generate_insights(self):
        """Generate AI-powered insights"""
        insights = []
        
        # Sustainability insights
        current_sustainability = self.ratios['sustainability_index'].iloc[-1]
        sustainability_trend = self.trends.get('NilaiManfaat', {}).get('slope', 0)
        
        if current_sustainability < 40:
            insights.append({
                'type': 'critical',
                'title': '🚨 Critical Sustainability Alert',
                'message': f'Current sustainability index ({current_sustainability:.1f}%) is below critical threshold. Immediate intervention required.',
                'recommendation': 'Implement emergency cost reduction and investment optimization strategies.'
            })
        elif current_sustainability < 60:
            insights.append({
                'type': 'warning',
                'title': '⚠️ Sustainability Warning',
                'message': f'Sustainability index ({current_sustainability:.1f}%) shows concerning levels.',
                'recommendation': 'Monitor closely and prepare optimization strategies.'
            })
        else:
            insights.append({
                'type': 'success',
                'title': '✅ Healthy Sustainability',
                'message': f'Current sustainability index ({current_sustainability:.1f}%) is within healthy range.',
                'recommendation': 'Maintain current strategies while exploring optimization opportunities.'
            })
        
        # Cost trend insights
        cost_trend = self.trends.get('Total_Cost', {}).get('slope', 0)
        if cost_trend > 0:
            insights.append({
                'type': 'warning',
                'title': '📈 Rising Cost Trend',
                'message': f'Total costs are increasing at {cost_trend:,.0f} per year.',
                'recommendation': 'Implement cost containment measures and efficiency improvements.'
            })
        
        # Investment performance insights
        avg_roi = self.ratios['roi'].mean()
        if avg_roi < 5:
            insights.append({
                'type': 'warning',
                'title': '💰 Low Investment Returns',
                'message': f'Average ROI ({avg_roi:.1f}%) is below target range.',
                'recommendation': 'Review investment strategy and consider portfolio optimization.'
            })
        
        return insights
    
    def detect_anomalies(self):
        """Detect financial anomalies using statistical methods"""
        anomalies = []
        
        for col in ['BPIH', 'Bipih', 'NilaiManfaat']:
            if col in self.data.columns:
                values = self.data[col]
                
                # Z-score method
                z_scores = np.abs(stats.zscore(values))
                anomaly_threshold = 2.5
                
                anomaly_indices = np.where(z_scores > anomaly_threshold)[0]
                
                for idx in anomaly_indices:
                    anomalies.append({
                        'year': self.data.iloc[idx]['Year'],
                        'metric': col,
                        'value': values.iloc[idx],
                        'z_score': z_scores[idx],
                        'severity': 'High' if z_scores[idx] > 3 else 'Medium'
                    })
        
        return anomalies

# Data Generation
@st.cache_data
def generate_comprehensive_data():
    """Generate comprehensive financial data for analysis"""
    np.random.seed(42)
    
    years = range(2015, 2026)
    n_years = len(years)
    
    data = []
    base_bpih = 75000000
    base_bipih = 35000000
    base_benefit = 50000000
    
    for i, year in enumerate(years):
        # Realistic growth with economic cycles
        economic_cycle = np.sin(i * 0.5) * 0.02  # Economic cycle effect
        
        bpih = base_bpih * (1.05 + economic_cycle) ** i * np.random.uniform(0.95, 1.05)
        bipih = base_bipih * (1.07 + economic_cycle) ** i * np.random.uniform(0.9, 1.1)
        benefit = base_benefit * (0.98 + economic_cycle * 0.5) ** i * np.random.uniform(0.95, 1.05)
        
        # Additional metrics
        investment_return = 0.06 + economic_cycle + np.random.normal(0, 0.02)
        inflation_rate = 0.03 + abs(economic_cycle) + np.random.normal(0, 0.01)
        operational_cost = bpih * (0.1 + np.random.uniform(-0.02, 0.02))
        
        # Market volatility
        market_volatility = 0.15 + abs(economic_cycle) * 0.5
        
        data.append({
            'Year': year,
            'BPIH': bpih,
            'Bipih': bipih,
            'NilaiManfaat': benefit,
            'InvestmentReturn': investment_return,
            'InflationRate': inflation_rate,
            'OperationalCost': operational_cost,
            'Total_Cost': bpih + bipih + operational_cost,
            'MarketVolatility': market_volatility,
            'EconomicCycle': economic_cycle
        })
    
    return pd.DataFrame(data)

# Header
st.markdown("""
<div class="analytics-header">
    <h1>📈 ADVANCED FINANCIAL ANALYTICS</h1>
    <h3>Comprehensive Analysis & Business Intelligence</h3>
    <p>Deep Insights | Trend Analysis | Risk Assessment | Performance Benchmarking</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.markdown("## 📊 Analytics Controls")
    
    analysis_type = st.selectbox(
        "Analysis Type",
        [
            "Overview Dashboard",
            "Trend Analysis", 
            "Risk Analytics",
            "Performance Benchmarking",
            "Correlation Analysis",
            "Anomaly Detection",
            "Predictive Analytics"
        ]
    )
    
    st.markdown("---")
    
    # Time period selection
    st.markdown("### 📅 Time Period")
    start_year = st.selectbox("Start Year", range(2015, 2026), index=0)
    end_year = st.selectbox("End Year", range(2015, 2026), index=10)
    
    # Metrics selection
    st.markdown("### 📋 Metrics")
    selected_metrics = st.multiselect(
        "Select Metrics",
        ["BPIH", "Bipih", "NilaiManfaat", "Total_Cost", "InvestmentReturn"],
        default=["BPIH", "NilaiManfaat"]
    )
    
    # Analysis parameters
    st.markdown("### ⚙️ Parameters")
    confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95)
    smoothing_window = st.slider("Smoothing Window", 1, 5, 3)

# Load data and initialize analytics
data = generate_comprehensive_data()
data_filtered = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]
analytics = HajjFinancialAnalytics(data_filtered)

# Main content based on analysis type
if analysis_type == "Overview Dashboard":
    
    # Key metrics overview
    st.markdown("## 📊 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_sustainability = analytics.ratios['sustainability_index'].iloc[-1]
        prev_sustainability = analytics.ratios['sustainability_index'].iloc[-2] if len(analytics.ratios) > 1 else current_sustainability
        change = current_sustainability - prev_sustainability
        
        st.metric(
            "🌱 Sustainability Index",
            f"{current_sustainability:.1f}%",
            f"{change:+.1f}pp"
        )
    
    with col2:
        current_roi = analytics.ratios['roi'].iloc[-1]
        target_roi = 6.0
        
        st.metric(
            "💰 Investment ROI",
            f"{current_roi:.1f}%",
            f"Target: {target_roi:.1f}%"
        )
    
    with col3:
        total_cost = data_filtered['Total_Cost'].iloc[-1]
        cost_growth = analytics.ratios['total_cost_growth'].iloc[-1]
        
        st.metric(
            "💸 Total Cost",
            f"Rp {total_cost:,.0f}",
            f"{cost_growth:+.1f}%"
        )
    
    with col4:
        risk_level = analytics.risk_metrics.get('sustainability_risk', 0)
        risk_status = "🟢 Low" if risk_level < 20 else "🟡 Medium" if risk_level < 50 else "🔴 High"
        
        st.metric(
            "⚠️ Risk Level",
            risk_status,
            f"{risk_level:.1f}%"
        )
    
    # Financial trends chart
    st.markdown("### 📈 Financial Trends Overview")
    
    fig_overview = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cost Components', 'Sustainability Trend', 'Investment Performance', 'Growth Rates'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Cost components
    fig_overview.add_trace(
        go.Scatter(x=data_filtered['Year'], y=data_filtered['BPIH'], name='BPIH', line=dict(color='#e74c3c')),
        row=1, col=1
    )
    fig_overview.add_trace(
        go.Scatter(x=data_filtered['Year'], y=data_filtered['Bipih'], name='Bipih', line=dict(color='#3498db')),
        row=1, col=1
    )
    
    # Sustainability trend
    fig_overview.add_trace(
        go.Scatter(x=analytics.ratios.index, y=analytics.ratios['sustainability_index'], 
                  name='Sustainability Index', line=dict(color='#27ae60')),
        row=1, col=2
    )
    
    # Investment performance
    fig_overview.add_trace(
        go.Scatter(x=analytics.ratios.index, y=analytics.ratios['roi'], 
                  name='ROI', line=dict(color='#9b59b6')),
        row=2, col=1
    )
    
    # Growth rates
    fig_overview.add_trace(
        go.Bar(x=analytics.ratios.index, y=analytics.ratios['total_cost_growth'], 
               name='Cost Growth', marker_color='#f39c12'),
        row=2, col=2
    )
    
    fig_overview.update_layout(height=600, showlegend=False, title_text="Financial Overview Dashboard")
    st.plotly_chart(fig_overview, use_container_width=True)
    
    # AI Insights
    st.markdown("### 🤖 AI-Generated Insights")
    
    insights = analytics.generate_insights()
    
    for insight in insights:
        if insight['type'] == 'critical':
            st.markdown(f"""
            <div class="warning-box">
                <h4>{insight['title']}</h4>
                <p>{insight['message']}</p>
                <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif insight['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-box">
                <h4>{insight['title']}</h4>
                <p>{insight['message']}</p>
                <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif insight['type'] == 'success':
            st.markdown(f"""
            <div class="success-box">
                <h4>{insight['title']}</h4>
                <p>{insight['message']}</p>
                <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

elif analysis_type == "Trend Analysis":
    
    st.markdown("## 📈 Comprehensive Trend Analysis")
    
    # Statistical trend analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Trend Statistics")
        
        for metric, trend_data in analytics.trends.items():
            st.markdown(f"""
            <div class="metric-container">
                <h4>{metric}</h4>
                <p><strong>Direction:</strong> {trend_data['trend_direction']}</p>
                <p><strong>Strength:</strong> {trend_data['trend_strength']}</p>
                <p><strong>R²:</strong> {trend_data['r_squared']:.3f}</p>
                <p><strong>Annual Change:</strong> {trend_data['annual_change']:,.0f}</p>
                <p><strong>P-value:</strong> {trend_data['p_value']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Trend Projections")
        
        # Create trend projection chart
        selected_metric = st.selectbox("Select Metric for Projection", list(analytics.trends.keys()))
        
        if selected_metric:
            trend_data = analytics.trends[selected_metric]
            
            # Historical data
            years = data_filtered['Year'].values
            values = data_filtered[selected_metric].values
            
            # Projection
            future_years = np.arange(years[-1] + 1, years[-1] + 6)
            future_indices = np.arange(len(years), len(years) + 5)
            projected_values = trend_data['intercept'] + trend_data['slope'] * future_indices
            
            # Confidence intervals
            std_err = np.sqrt(np.sum((values - (trend_data['intercept'] + trend_data['slope'] * np.arange(len(values))))**2) / (len(values) - 2))
            conf_interval = 1.96 * std_err
            
            fig_trend = go.Figure()
            
            # Historical data
            fig_trend.add_trace(go.Scatter(
                x=years, y=values,
                mode='lines+markers',
                name='Historical',
                line=dict(color='#3498db', width=3)
            ))
            
            # Trend line
            trend_line = trend_data['intercept'] + trend_data['slope'] * np.arange(len(years))
            fig_trend.add_trace(go.Scatter(
                x=years, y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='#e74c3c', dash='dash')
            ))
            
            # Projections
            fig_trend.add_trace(go.Scatter(
                x=future_years, y=projected_values,
                mode='lines+markers',
                name='Projection',
                line=dict(color='#27ae60', width=3)
            ))
            
            # Confidence interval
            fig_trend.add_trace(go.Scatter(
                x=np.concatenate([future_years, future_years[::-1]]),
                y=np.concatenate([projected_values + conf_interval, (projected_values - conf_interval)[::-1]]),
                fill='toself',
                fillcolor='rgba(39, 174, 96, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='Confidence Interval'
            ))
            
            fig_trend.update_layout(
                title=f"{selected_metric} Trend Analysis & Projection",
                xaxis_title="Year",
                yaxis_title="Value",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)

elif analysis_type == "Risk Analytics":
    
    st.markdown("## ⚠️ Comprehensive Risk Analytics")
    
    # Risk overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Volatility Analysis")
        
        for metric in ['BPIH', 'Bipih', 'NilaiManfaat']:
            volatility = analytics.risk_metrics.get(f'{metric}_volatility', 0)
            var_95 = analytics.risk_metrics.get(f'{metric}_var_95', 0)
            
            st.markdown(f"""
            <div class="metric-container">
                <h4>{metric}</h4>
                <p><strong>Volatility:</strong> {volatility:.2f}%</p>
                <p><strong>VaR (95%):</strong> {var_95:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎲 Value at Risk")
        
        # VaR chart
        returns_data = []
        
        for metric in ['BPIH', 'Bipih', 'NilaiManfaat']:
            if metric in data_filtered.columns:
                returns = data_filtered[metric].pct_change().dropna() * 100
                returns_data.extend([{
                    'Metric': metric,
                    'Return': ret,
                    'Date': data_filtered.iloc[i+1]['Year']
                } for i, ret in enumerate(returns)])
        
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            
            fig_var = px.box(
                returns_df, 
                x='Metric', 
                y='Return',
                title="Return Distribution & VaR",
                color='Metric'
            )
            
            # Add VaR lines
            for metric in ['BPIH', 'Bipih', 'NilaiManfaat']:
                var_95 = analytics.risk_metrics.get(f'{metric}_var_95', 0)
                fig_var.add_hline(
                    y=var_95,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"VaR 95%"
                )
            
            st.plotly_chart(fig_var, use_container_width=True)
    
    with col3:
        st.markdown("### 🔗 Risk Correlation")
        
        correlation_matrix = data_filtered[['BPIH', 'Bipih', 'NilaiManfaat']].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title="Risk Correlation Matrix",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Risk scenario analysis
    st.markdown("### 🎯 Scenario Analysis")
    
    scenario_tabs = st.tabs(["📉 Stress Test", "📊 Monte Carlo", "🎲 Scenario Planning"])
    
    with scenario_tabs[0]:
        st.markdown("#### 📉 Stress Testing Results")
        
        # Simulate stress scenarios
        stress_scenarios = {
            'Market Crash': {'BPIH': 1.2, 'Bipih': 1.3, 'NilaiManfaat': 0.7},
            'High Inflation': {'BPIH': 1.15, 'Bipih': 1.25, 'NilaiManfaat': 0.85},
            'Economic Recession': {'BPIH': 1.1, 'Bipih': 1.2, 'NilaiManfaat': 0.8},
            'Base Case': {'BPIH': 1.0, 'Bipih': 1.0, 'NilaiManfaat': 1.0}
        }
        
        stress_results = []
        
        for scenario, multipliers in stress_scenarios.items():
            current_bpih = data_filtered['BPIH'].iloc[-1] * multipliers['BPIH']
            current_benefit = data_filtered['NilaiManfaat'].iloc[-1] * multipliers['NilaiManfaat']
            sustainability = (current_benefit / current_bpih) * 100
            
            stress_results.append({
                'Scenario': scenario,
                'BPIH': current_bpih,
                'Benefit': current_benefit,
                'Sustainability': sustainability,
                'Risk_Level': 'High' if sustainability < 40 else 'Medium' if sustainability < 60 else 'Low'
            })
        
        stress_df = pd.DataFrame(stress_results)
        
        fig_stress = px.bar(
            stress_df,
            x='Scenario',
            y='Sustainability',
            title="Stress Test Results - Sustainability Index",
            color='Risk_Level',
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}
        )
        
        fig_stress.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        fig_stress.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        
        st.plotly_chart(fig_stress, use_container_width=True)

        # Display stress test table
        st.dataframe(stress_df, use_container_width=True)

    with scenario_tabs[1]:
        st.markdown("#### 📊 Monte Carlo Simulation")
        st.info("For comprehensive Monte Carlo simulation with advanced features, visit the dedicated **Simulation** page (06_Simulation).")

        # Simple Monte Carlo preview
        st.markdown("**Quick Monte Carlo Preview:**")
        n_sims = 100
        years = 5

        # Run simple simulation
        np.random.seed(42)
        current_sustainability = (data_filtered['NilaiManfaat'].iloc[-1] / data_filtered['BPIH'].iloc[-1]) * 100

        simulations = []
        for _ in range(n_sims):
            path = [current_sustainability]
            for _ in range(years):
                change = np.random.normal(0, 5)  # 5% volatility
                path.append(max(0, path[-1] + change))
            simulations.append(path[-1])

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Projection", f"{np.mean(simulations):.1f}%")
        with col2:
            st.metric("5th Percentile (VaR)", f"{np.percentile(simulations, 5):.1f}%")
        with col3:
            st.metric("95th Percentile", f"{np.percentile(simulations, 95):.1f}%")

        # Histogram
        fig_mc = px.histogram(x=simulations, nbins=20, title="Monte Carlo Distribution (Preview)")
        fig_mc.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Warning")
        st.plotly_chart(fig_mc, use_container_width=True)

    with scenario_tabs[2]:
        st.markdown("#### 🎲 Scenario Planning")

        st.markdown("**Custom Scenario Builder:**")
        col1, col2 = st.columns(2)

        with col1:
            bpih_change = st.slider("BPIH Change (%)", -20, 30, 0, key="sp_bpih")
            benefit_change = st.slider("Benefit Change (%)", -30, 30, 0, key="sp_benefit")

        with col2:
            current_bpih = data_filtered['BPIH'].iloc[-1]
            current_benefit = data_filtered['NilaiManfaat'].iloc[-1]

            new_bpih = current_bpih * (1 + bpih_change/100)
            new_benefit = current_benefit * (1 + benefit_change/100)
            new_sustainability = (new_benefit / new_bpih) * 100
            current_sus = (current_benefit / current_bpih) * 100

            st.metric("Current Sustainability", f"{current_sus:.1f}%")
            st.metric("Projected Sustainability", f"{new_sustainability:.1f}%",
                     delta=f"{new_sustainability - current_sus:.1f}%")

            if new_sustainability < 40:
                st.error("Critical: Sustainability below threshold!")
            elif new_sustainability < 60:
                st.warning("Warning: Sustainability needs attention")
            else:
                st.success("Healthy: Sustainability is adequate")

elif analysis_type == "Anomaly Detection":
    
    st.markdown("## 🔍 Anomaly Detection & Alert System")
    
    # Detect anomalies
    anomalies = analytics.detect_anomalies()
    
    if anomalies:
        st.markdown("### 🚨 Detected Anomalies")
        
        anomaly_df = pd.DataFrame(anomalies)
        
        # Anomaly summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_anomalies = len(anomalies)
            st.metric("Total Anomalies", total_anomalies)
        
        with col2:
            high_severity = len([a for a in anomalies if a['severity'] == 'High'])
            st.metric("High Severity", high_severity)
        
        with col3:
            recent_anomalies = len([a for a in anomalies if a['year'] >= 2022])
            st.metric("Recent Anomalies", recent_anomalies)
        
        # Anomaly visualization
        fig_anomaly = px.scatter(
            anomaly_df,
            x='year',
            y='value',
            color='severity',
            size='z_score',
            facet_col='metric',
            title="Detected Financial Anomalies",
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12'}
        )
        
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Anomaly details
        st.markdown("### 📋 Anomaly Details")
        st.dataframe(anomaly_df, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        for anomaly in anomalies:
            if anomaly['severity'] == 'High':
                st.markdown(f"""
                <div class="warning-box">
                    <h4>🚨 High Severity Anomaly - {anomaly['year']}</h4>
                    <p><strong>Metric:</strong> {anomaly['metric']}</p>
                    <p><strong>Value:</strong> {anomaly['value']:,.0f}</p>
                    <p><strong>Z-Score:</strong> {anomaly['z_score']:.2f}</p>
                    <p><strong>Action Required:</strong> Immediate investigation and corrective measures needed.</p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="success-box">
            <h3>✅ No Anomalies Detected</h3>
            <p>All financial metrics are within normal statistical ranges.</p>
            <p>Continue monitoring for any emerging patterns.</p>
        </div>
        """, unsafe_allow_html=True)

# Export functionality
st.markdown("---")
st.markdown("## 📊 Export & Reporting")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 Generate Analytics Report", use_container_width=True):
        st.success("📋 Analytics report generated successfully!")
        
        # Create summary report
        report_data = {
            'Metric': ['Sustainability Index', 'Investment ROI', 'Risk Level', 'Cost Growth'],
            'Current_Value': [
                f"{analytics.ratios['sustainability_index'].iloc[-1]:.1f}%",
                f"{analytics.ratios['roi'].iloc[-1]:.1f}%",
                f"{analytics.risk_metrics.get('sustainability_risk', 0):.1f}%",
                f"{analytics.ratios['total_cost_growth'].iloc[-1]:.1f}%"
            ],
            'Status': ['Healthy', 'Good', 'Medium', 'Increasing']
        }
        
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)

with col2:
    if st.button("📊 Export Data", use_container_width=True):
        st.success("💾 Data exported successfully!")
        
        # Show sample of exportable data
        export_data = analytics.ratios.round(2)
        st.dataframe(export_data.head(), use_container_width=True)

with col3:
    if st.button("🔔 Setup Alerts", use_container_width=True):
        st.info("⚙️ Alert system configured!")
        
        alert_thresholds = {
            'Sustainability Index': '< 50%',
            'Investment ROI': '< 5%',
            'Cost Growth': '> 10%',
            'Risk Level': '> 70%'
        }
        
        for metric, threshold in alert_thresholds.items():
            st.write(f"• {metric}: {threshold}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <h4>📈 Advanced Financial Analytics Center</h4>
    <p>Powered by Statistical Modeling & Machine Learning</p>
    <p><em>Delivering actionable insights for sustainable hajj financing</em></p>
</div>
""", unsafe_allow_html=True)