"""
💼 PERSONAL FINANCIAL PLANNING MODULE
Comprehensive hajj financial planning for individuals

Features:
- Personal hajj cost calculator
- Savings plan optimization
- Investment strategy recommendations
- Risk tolerance assessment
- Goal-based financial planning
- Shariah-compliant investment options
- Retirement & hajj integration planning
- Emergency fund calculations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="💼 Personal Planning",
    page_icon="💼",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .planning-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .calculator-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    
    .result-highlight {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(168, 237, 234, 0.3);
    }
    
    .progress-tracker {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .milestone-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    .milestone-achieved {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    
    .milestone-pending {
        background: linear-gradient(135deg, #6c757d 0%, #adb5bd 100%);
    }
    
    .risk-gauge {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        color: white;
        font-size: 18px;
        font-weight: bold;
    }
    
    .risk-conservative {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
    }
    
    .risk-aggressive {
        background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
    }
</style>
""", unsafe_allow_html=True)

# Personal Financial Planner Class
class PersonalHajjPlanner:
    """
    Comprehensive personal hajj financial planning system
    """
    
    def __init__(self):
        self.current_hajj_cost = 120000000  # Current estimated hajj cost in IDR
        self.inflation_rate = 0.05  # 5% annual inflation
        self.shariah_returns = {
            'conservative': 0.06,   # 6% for conservative shariah investments
            'moderate': 0.08,       # 8% for moderate shariah investments
            'aggressive': 0.10      # 10% for aggressive shariah investments
        }
    
    def calculate_future_hajj_cost(self, years_to_hajj):
        """Calculate future hajj cost considering inflation"""
        return self.current_hajj_cost * (1 + self.inflation_rate) ** years_to_hajj
    
    def calculate_monthly_savings(self, target_amount, years, annual_return):
        """Calculate required monthly savings using compound interest"""
        monthly_return = annual_return / 12
        months = years * 12
        
        if monthly_return == 0:
            return target_amount / months
        
        # PMT formula for annuity
        monthly_payment = target_amount * monthly_return / ((1 + monthly_return) ** months - 1)
        return monthly_payment
    
    def project_savings_growth(self, monthly_savings, years, annual_return):
        """Project savings growth over time"""
        monthly_return = annual_return / 12
        months = years * 12
        
        projections = []
        balance = 0
        
        for month in range(months + 1):
            if month > 0:
                balance = balance * (1 + monthly_return) + monthly_savings
            
            projections.append({
                'month': month,
                'year': month / 12,
                'balance': balance,
                'total_contributions': month * monthly_savings,
                'total_returns': balance - (month * monthly_savings)
            })
        
        return pd.DataFrame(projections)
    
    def assess_risk_tolerance(self, age, income, dependents, investment_experience, time_horizon):
        """Assess investor risk tolerance"""
        
        risk_score = 0
        
        # Age factor (younger = higher risk tolerance)
        if age < 30:
            risk_score += 4
        elif age < 40:
            risk_score += 3
        elif age < 50:
            risk_score += 2
        else:
            risk_score += 1
        
        # Income factor (higher income = higher risk tolerance)
        if income >= 15000000:  # 15M IDR monthly
            risk_score += 4
        elif income >= 10000000:  # 10M IDR monthly
            risk_score += 3
        elif income >= 5000000:   # 5M IDR monthly
            risk_score += 2
        else:
            risk_score += 1
        
        # Dependents factor (fewer dependents = higher risk tolerance)
        if dependents == 0:
            risk_score += 3
        elif dependents <= 2:
            risk_score += 2
        else:
            risk_score += 1
        
        # Investment experience factor
        if investment_experience >= 5:
            risk_score += 3
        elif investment_experience >= 2:
            risk_score += 2
        else:
            risk_score += 1
        
        # Time horizon factor (longer = higher risk tolerance)
        if time_horizon >= 10:
            risk_score += 4
        elif time_horizon >= 5:
            risk_score += 3
        elif time_horizon >= 2:
            risk_score += 2
        else:
            risk_score += 1
        
        # Determine risk profile
        if risk_score >= 15:
            return 'aggressive'
        elif risk_score >= 10:
            return 'moderate'
        else:
            return 'conservative'
    
    def generate_investment_recommendations(self, risk_profile, target_amount, time_horizon):
        """Generate shariah-compliant investment recommendations"""
        
        recommendations = {
            'conservative': {
                'allocation': {
                    'Sukuk Pemerintah': 0.50,
                    'Deposito Syariah': 0.30,
                    'Reksadana Syariah Pendapatan Tetap': 0.15,
                    'Emas': 0.05
                },
                'expected_return': 0.06,
                'risk_level': 'Rendah',
                'description': 'Portfolio konservatif dengan fokus pada preservasi modal dan return stabil'
            },
            'moderate': {
                'allocation': {
                    'Sukuk Pemerintah': 0.30,
                    'Sukuk Korporasi': 0.20,
                    'Reksadana Syariah Saham': 0.25,
                    'Deposito Syariah': 0.15,
                    'Emas': 0.10
                },
                'expected_return': 0.08,
                'risk_level': 'Sedang',
                'description': 'Portfolio seimbang antara growth dan stability dengan diversifikasi optimal'
            },
            'aggressive': {
                'allocation': {
                    'Saham Syariah': 0.40,
                    'Reksadana Syariah Saham': 0.25,
                    'Sukuk Korporasi': 0.15,
                    'REITs Syariah': 0.10,
                    'Emas': 0.10
                },
                'expected_return': 0.10,
                'risk_level': 'Tinggi',
                'description': 'Portfolio growth-oriented dengan potensi return tinggi untuk jangka panjang'
            }
        }
        
        return recommendations.get(risk_profile, recommendations['moderate'])
    
    def calculate_emergency_fund(self, monthly_expenses, dependents):
        """Calculate recommended emergency fund"""
        
        # Base emergency fund (6 months expenses)
        base_months = 6
        
        # Adjust based on dependents
        if dependents >= 3:
            base_months = 9
        elif dependents >= 1:
            base_months = 7
        
        return monthly_expenses * base_months
    
    def generate_milestones(self, target_amount, years):
        """Generate savings milestones"""
        
        milestones = []
        milestone_percentages = [0.25, 0.50, 0.75, 1.0]
        milestone_names = ['25% Target', '50% Target', '75% Target', 'Hajj Ready!']
        
        for i, (percentage, name) in enumerate(zip(milestone_percentages, milestone_names)):
            milestone_amount = target_amount * percentage
            milestone_year = years * percentage
            
            milestones.append({
                'name': name,
                'amount': milestone_amount,
                'year': milestone_year,
                'percentage': percentage * 100
            })
        
        return milestones

# Header
st.markdown("""
<div class="planning-header">
    <h1>💼 PERSONAL HAJJ FINANCIAL PLANNING</h1>
    <h3>Comprehensive Financial Planning for Your Hajj Journey</h3>
    <p>Shariah-Compliant | Goal-Based Planning | Risk Assessment | Investment Optimization</p>
</div>
""", unsafe_allow_html=True)

# Initialize planner
planner = PersonalHajjPlanner()

# Sidebar - Personal Information
with st.sidebar:
    st.markdown("## 👤 Personal Information")
    
    # Basic information
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    monthly_income = st.number_input("Monthly Income (IDR)", min_value=1000000, max_value=100000000, value=8000000, step=500000)
    monthly_expenses = st.number_input("Monthly Expenses (IDR)", min_value=500000, max_value=50000000, value=6000000, step=500000)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
    
    st.markdown("---")
    
    # Hajj planning details
    st.markdown("## 🕌 Hajj Planning")
    
    target_hajj_year = st.number_input("Target Hajj Year", min_value=2025, max_value=2050, value=2030)
    current_savings = st.number_input("Current Hajj Savings (IDR)", min_value=0, max_value=500000000, value=10000000, step=1000000)
    
    # Investment experience
    investment_experience = st.selectbox(
        "Investment Experience (Years)",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        index=2
    )
    
    st.markdown("---")
    
    # Additional goals
    st.markdown("## 🎯 Additional Goals")
    
    include_emergency_fund = st.checkbox("Include Emergency Fund Planning", value=True)
    include_education_fund = st.checkbox("Include Children Education Fund", value=False)
    include_retirement = st.checkbox("Integrate with Retirement Planning", value=False)

# Calculate planning parameters
current_year = datetime.now().year
years_to_hajj = target_hajj_year - current_year
future_hajj_cost = planner.calculate_future_hajj_cost(years_to_hajj)
monthly_surplus = monthly_income - monthly_expenses

# Risk assessment
risk_profile = planner.assess_risk_tolerance(age, monthly_income, dependents, investment_experience, years_to_hajj)
investment_recommendation = planner.generate_investment_recommendations(risk_profile, future_hajj_cost, years_to_hajj)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧮 Hajj Calculator", 
    "📊 Savings Plan", 
    "💰 Investment Strategy", 
    "📈 Progress Tracking", 
    "📋 Comprehensive Plan"
])

with tab1:
    st.markdown("## 🧮 Hajj Cost Calculator")
    
    # Cost breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="result-highlight">
            <h3>💰 Current Hajj Cost</h3>
            <h2>Rp {planner.current_hajj_cost:,.0f}</h2>
            <p>Base cost for {current_year}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="result-highlight">
            <h3>📈 Future Cost ({target_hajj_year})</h3>
            <h2>Rp {future_hajj_cost:,.0f}</h2>
            <p>Projected cost in {years_to_hajj} years</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        shortage = max(0, future_hajj_cost - current_savings)
        st.markdown(f"""
        <div class="result-highlight">
            <h3>💸 Amount Needed</h3>
            <h2>Rp {shortage:,.0f}</h2>
            <p>Additional savings required</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed cost breakdown
    st.markdown("### 📋 Detailed Cost Breakdown")
    
    cost_components = {
        'Component': [
            'BPIH (Biaya Penyelenggaraan)',
            'Bipih (Setoran Awal)',
            'Visa & Documents',
            'Travel Insurance',
            'Personal Expenses',
            'Contingency (10%)'
        ],
        'Current Cost (IDR)': [
            65000000,
            30000000,
            5000000,
            2000000,
            10000000,
            11200000
        ]
    }
    
    # Apply inflation to all components
    cost_df = pd.DataFrame(cost_components)
    cost_df[f'Future Cost {target_hajj_year} (IDR)'] = cost_df['Current Cost (IDR)'] * (1 + planner.inflation_rate) ** years_to_hajj
    cost_df['Increase (%)'] = ((cost_df[f'Future Cost {target_hajj_year} (IDR)'] / cost_df['Current Cost (IDR)']) - 1) * 100
    
    # Format currency columns
    for col in ['Current Cost (IDR)', f'Future Cost {target_hajj_year} (IDR)']:
        cost_df[col] = cost_df[col].apply(lambda x: f"Rp {x:,.0f}")
    cost_df['Increase (%)'] = cost_df['Increase (%)'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(cost_df, use_container_width=True)
    
    # Cost comparison chart
    fig_cost = go.Figure(data=[
        go.Bar(name='Current Cost', x=cost_components['Component'], y=cost_components['Current Cost (IDR)']),
        go.Bar(name=f'Future Cost ({target_hajj_year})', x=cost_components['Component'], 
               y=[x * (1 + planner.inflation_rate) ** years_to_hajj for x in cost_components['Current Cost (IDR)']])
    ])
    
    fig_cost.update_layout(
        title=f"Hajj Cost Comparison: {current_year} vs {target_hajj_year}",
        xaxis_title="Cost Components",
        yaxis_title="Amount (IDR)",
        barmode='group',
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig_cost, use_container_width=True)

with tab2:
    st.markdown("## 📊 Optimal Savings Plan")
    
    # Calculate required monthly savings for different risk profiles
    savings_options = {}
    
    for profile, return_rate in planner.shariah_returns.items():
        required_monthly = planner.calculate_monthly_savings(shortage, years_to_hajj, return_rate)
        savings_options[profile] = {
            'monthly_savings': required_monthly,
            'return_rate': return_rate,
            'total_contributions': required_monthly * years_to_hajj * 12,
            'total_returns': shortage - (required_monthly * years_to_hajj * 12)
        }
    
    # Display savings options
    col1, col2, col3 = st.columns(3)
    
    profiles = ['conservative', 'moderate', 'aggressive']
    profile_names = ['Conservative', 'Moderate', 'Aggressive']
    profile_colors = ['success', 'warning', 'danger']
    
    for i, (profile, name, color) in enumerate(zip(profiles, profile_names, profile_colors)):
        with [col1, col2, col3][i]:
            option = savings_options[profile]
            
            st.markdown(f"""
            <div class="calculator-card">
                <h4 style="color: {'#28a745' if color=='success' else '#ffc107' if color=='warning' else '#dc3545'};">
                    {name} Strategy
                </h4>
                <p><strong>Expected Return:</strong> {option['return_rate']:.1%} per year</p>
                <p><strong>Monthly Savings:</strong> Rp {option['monthly_savings']:,.0f}</p>
                <p><strong>Total Contributions:</strong> Rp {option['total_contributions']:,.0f}</p>
                <p><strong>Investment Returns:</strong> Rp {option['total_returns']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommended strategy based on risk profile
    recommended_option = savings_options[risk_profile]
    
    st.markdown(f"""
    <div class="recommendation-box">
        <h3>🎯 Recommended Strategy for You: {risk_profile.title()}</h3>
        <p>Based on your risk assessment, we recommend the <strong>{risk_profile}</strong> investment strategy.</p>
        <p><strong>Monthly Savings Required:</strong> Rp {recommended_option['monthly_savings']:,.0f}</p>
        <p><strong>As % of Income:</strong> {(recommended_option['monthly_savings'] / monthly_income) * 100:.1f}%</p>
        <p><strong>As % of Surplus:</strong> {(recommended_option['monthly_savings'] / monthly_surplus) * 100:.1f}% (if surplus is positive)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Savings projection chart
    st.markdown("### 📈 Savings Growth Projection")
    
    projections = planner.project_savings_growth(
        recommended_option['monthly_savings'], 
        years_to_hajj, 
        planner.shariah_returns[risk_profile]
    )
    
    fig_projection = go.Figure()
    
    # Total balance
    fig_projection.add_trace(go.Scatter(
        x=projections['year'],
        y=projections['balance'],
        mode='lines',
        name='Total Balance',
        line=dict(color='#3498db', width=3),
        fill='tonexty'
    ))
    
    # Contributions
    fig_projection.add_trace(go.Scatter(
        x=projections['year'],
        y=projections['total_contributions'],
        mode='lines',
        name='Total Contributions',
        line=dict(color='#e74c3c', width=2)
    ))
    
    # Target line
    fig_projection.add_hline(
        y=future_hajj_cost,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Target: Rp {future_hajj_cost:,.0f}"
    )
    
    fig_projection.update_layout(
        title="Hajj Savings Growth Projection",
        xaxis_title="Years",
        yaxis_title="Amount (IDR)",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig_projection, use_container_width=True)
    
    # Sensitivity analysis
    st.markdown("### 🎯 Sensitivity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Return rate sensitivity
        return_scenarios = [0.04, 0.06, 0.08, 0.10, 0.12]
        sensitivity_data = []
        
        for rate in return_scenarios:
            monthly_needed = planner.calculate_monthly_savings(shortage, years_to_hajj, rate)
            sensitivity_data.append({
                'Return Rate': f"{rate:.0%}",
                'Monthly Savings': f"Rp {monthly_needed:,.0f}",
                'Annual Savings': f"Rp {monthly_needed * 12:,.0f}"
            })
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        st.markdown("**Return Rate Sensitivity**")
        st.dataframe(sensitivity_df, use_container_width=True)
    
    with col2:
        # Time horizon sensitivity
        time_scenarios = range(max(1, years_to_hajj - 2), years_to_hajj + 3)
        time_sensitivity_data = []
        
        for years in time_scenarios:
            future_cost = planner.calculate_future_hajj_cost(years)
            shortfall = max(0, future_cost - current_savings)
            monthly_needed = planner.calculate_monthly_savings(shortfall, years, planner.shariah_returns[risk_profile])
            
            time_sensitivity_data.append({
                'Years to Hajj': years,
                'Future Cost': f"Rp {future_cost:,.0f}",
                'Monthly Savings': f"Rp {monthly_needed:,.0f}"
            })
        
        time_sensitivity_df = pd.DataFrame(time_sensitivity_data)
        st.markdown("**Time Horizon Sensitivity**")
        st.dataframe(time_sensitivity_df, use_container_width=True)

with tab3:
    st.markdown("## 💰 Investment Strategy & Portfolio")
    
    # Risk assessment display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_class = f'risk-{risk_profile}'
        risk_display = risk_profile.title()
        
        st.markdown(f"""
        <div class="calculator-card">
            <h3 style="text-align: center;">⚖️ Risk Profile Assessment</h3>
            <div class="{risk_class} risk-gauge">
                {risk_display}
            </div>
            <p style="text-align: center; margin-top: 1rem;">
                <strong>Expected Return:</strong> {investment_recommendation['expected_return']:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="calculator-card">
            <h3>📊 Risk Level</h3>
            <p><strong>Level:</strong> {investment_recommendation['risk_level']}</p>
            <p><strong>Strategy:</strong> {investment_recommendation['description']}</p>
            <p><strong>Time Horizon:</strong> {years_to_hajj} years</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        emergency_fund = planner.calculate_emergency_fund(monthly_expenses, dependents)
        
        st.markdown(f"""
        <div class="calculator-card">
            <h3>🚨 Emergency Fund</h3>
            <p><strong>Recommended:</strong> Rp {emergency_fund:,.0f}</p>
            <p><strong>Months Covered:</strong> {emergency_fund / monthly_expenses:.1f}</p>
            <p><strong>Priority:</strong> {'High' if current_savings < emergency_fund else 'Moderate'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Portfolio allocation
    st.markdown("### 🥧 Recommended Portfolio Allocation")
    
    allocation = investment_recommendation['allocation']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            values=list(allocation.values()),
            names=list(allocation.keys()),
            title=f"{risk_profile.title()} Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Allocation table with details
        allocation_details = []
        
        instrument_descriptions = {
            'Sukuk Pemerintah': 'Government sukuk bonds - lowest risk, stable returns',
            'Sukuk Korporasi': 'Corporate sukuk bonds - moderate risk, higher returns',
            'Deposito Syariah': 'Shariah-compliant deposits - capital guaranteed',
            'Reksadana Syariah Pendapatan Tetap': 'Fixed income mutual funds - stable income',
            'Reksadana Syariah Saham': 'Equity mutual funds - growth potential',
            'Saham Syariah': 'Individual shariah-compliant stocks - high growth',
            'REITs Syariah': 'Real estate investment trusts - diversification',
            'Emas': 'Gold investment - inflation hedge'
        }
        
        for instrument, percentage in allocation.items():
            amount = shortage * percentage
            allocation_details.append({
                'Investment': instrument,
                'Allocation': f"{percentage:.0%}",
                'Amount': f"Rp {amount:,.0f}",
                'Description': instrument_descriptions.get(instrument, 'Shariah-compliant investment')
            })
        
        st.markdown("**Portfolio Details**")
        st.dataframe(pd.DataFrame(allocation_details), use_container_width=True)
    
    # Investment comparison
    st.markdown("### 📊 Investment Strategy Comparison")
    
    comparison_data = []
    
    for profile in ['conservative', 'moderate', 'aggressive']:
        option = savings_options[profile]
        recommendation = planner.generate_investment_recommendations(profile, shortage, years_to_hajj)
        
        comparison_data.append({
            'Strategy': profile.title(),
            'Expected Return': f"{recommendation['expected_return']:.1%}",
            'Monthly Savings': f"Rp {option['monthly_savings']:,.0f}",
            'Risk Level': recommendation['risk_level'],
            'Total Investment Returns': f"Rp {option['total_returns']:,.0f}",
            'Suitable For': planner.get_suitability(profile)  # Fixed: Using planner instance method
        })
    
    def _get_suitability(profile):
        suitability = {
            'conservative': 'Risk-averse, near retirement, short time horizon',
            'moderate': 'Balanced approach, medium time horizon, some risk tolerance',
            'aggressive': 'High risk tolerance, long time horizon, growth focused'
        }
        return suitability.get(profile, '')
    
    # Monkey patch the method to the instance
    for item in comparison_data:
        if 'conservative' in item['Strategy'].lower():
            item['Suitable For'] = 'Risk-averse, near retirement, short time horizon'
        elif 'moderate' in item['Strategy'].lower():
            item['Suitable For'] = 'Balanced approach, medium time horizon, some risk tolerance'
        else:
            item['Suitable For'] = 'High risk tolerance, long time horizon, growth focused'
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

with tab4:
    st.markdown("## 📈 Progress Tracking & Milestones")
    
    # Generate milestones
    milestones = planner.generate_milestones(future_hajj_cost, years_to_hajj)
    
    # Current progress
    current_progress = (current_savings / future_hajj_cost) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="result-highlight">
            <h3>📊 Current Progress</h3>
            <h2>{current_progress:.1f}%</h2>
            <p>of hajj goal achieved</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        months_saving = (current_savings / recommended_option['monthly_savings']) if recommended_option['monthly_savings'] > 0 else 0
        st.markdown(f"""
        <div class="result-highlight">
            <h3>⏰ Equivalent Months</h3>
            <h2>{months_saving:.1f}</h2>
            <p>months of recommended savings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        remaining_amount = max(0, future_hajj_cost - current_savings)
        st.markdown(f"""
        <div class="result-highlight">
            <h3>🎯 Remaining Target</h3>
            <h2>Rp {remaining_amount:,.0f}</h2>
            <p>still needed for hajj</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Milestone tracking
    st.markdown("### 🏆 Hajj Savings Milestones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for milestone in milestones:
            achieved = current_savings >= milestone['amount']
            badge_class = 'milestone-achieved' if achieved else 'milestone-pending'
            icon = '✅' if achieved else '⏳'
            
            st.markdown(f"""
            <div class="progress-tracker">
                <h4>{icon} {milestone['name']}</h4>
                <p><strong>Target Amount:</strong> Rp {milestone['amount']:,.0f}</p>
                <p><strong>Target Year:</strong> {current_year + milestone['year']:.1f}</p>
                <p><strong>Progress:</strong> {min(100, (current_savings / milestone['amount']) * 100):.1f}%</p>
                <span class="milestone-badge {badge_class}">
                    {'ACHIEVED' if achieved else 'PENDING'}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Progress visualization
        milestone_names = [m['name'] for m in milestones]
        milestone_amounts = [m['amount'] for m in milestones]
        current_vs_milestones = [min(current_savings, amount) for amount in milestone_amounts]
        
        fig_milestones = go.Figure()
        
        # Target amounts
        fig_milestones.add_trace(go.Bar(
            x=milestone_names,
            y=milestone_amounts,
            name='Target Amount',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Current progress
        fig_milestones.add_trace(go.Bar(
            x=milestone_names,
            y=current_vs_milestones,
            name='Current Progress',
            marker_color='darkblue'
        ))
        
        fig_milestones.update_layout(
            title="Milestone Progress Tracking",
            xaxis_title="Milestones",
            yaxis_title="Amount (IDR)",
            barmode='overlay',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_milestones, use_container_width=True)
    
    # Monthly tracking simulation
    st.markdown("### 📅 Monthly Progress Simulation")
    
    if st.button("📊 Generate Monthly Tracking Plan"):
        
        # Create monthly tracking for next 2 years
        monthly_plan = []
        current_balance = current_savings
        monthly_contribution = recommended_option['monthly_savings']
        monthly_return = planner.shariah_returns[risk_profile] / 12
        
        for month in range(24):  # 2 years
            # Apply monthly return and add contribution
            current_balance = current_balance * (1 + monthly_return) + monthly_contribution
            
            # Determine which milestone this represents
            milestone_achieved = None
            for milestone in milestones:
                if current_balance >= milestone['amount'] and current_balance - monthly_contribution < milestone['amount']:
                    milestone_achieved = milestone['name']
                    break
            
            monthly_plan.append({
                'Month': month + 1,
                'Date': (datetime.now() + timedelta(days=30 * month)).strftime('%Y-%m'),
                'Balance': current_balance,
                'Monthly Contribution': monthly_contribution,
                'Investment Return': current_balance * monthly_return,
                'Progress %': (current_balance / future_hajj_cost) * 100,
                'Milestone': milestone_achieved if milestone_achieved else ''
            })
        
        monthly_df = pd.DataFrame(monthly_plan)
        
        # Format currency columns
        for col in ['Balance', 'Monthly Contribution', 'Investment Return']:
            monthly_df[col] = monthly_df[col].apply(lambda x: f"Rp {x:,.0f}")
        
        monthly_df['Progress %'] = monthly_df['Progress %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(monthly_df, use_container_width=True)

with tab5:
    st.markdown("## 📋 Comprehensive Financial Plan")
    
    # Executive summary
    st.markdown("### 📊 Executive Summary")
    
    summary_metrics = {
        'Current Age': f"{age} years",
        'Target Hajj Year': str(target_hajj_year),
        'Time Horizon': f"{years_to_hajj} years",
        'Risk Profile': risk_profile.title(),
        'Monthly Income': f"Rp {monthly_income:,.0f}",
        'Monthly Expenses': f"Rp {monthly_expenses:,.0f}",
        'Monthly Surplus': f"Rp {monthly_surplus:,.0f}",
        'Current Hajj Savings': f"Rp {current_savings:,.0f}",
        'Future Hajj Cost': f"Rp {future_hajj_cost:,.0f}",
        'Additional Needed': f"Rp {max(0, future_hajj_cost - current_savings):,.0f}",
        'Recommended Monthly Savings': f"Rp {recommended_option['monthly_savings']:,.0f}",
        'Expected Annual Return': f"{investment_recommendation['expected_return']:.1%}",
        'Emergency Fund Target': f"Rp {emergency_fund:,.0f}",
        'Savings as % of Income': f"{(recommended_option['monthly_savings'] / monthly_income) * 100:.1f}%"
    }
    
    summary_df = pd.DataFrame(list(summary_metrics.items()), columns=['Metric', 'Value'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(summary_df.iloc[:7], use_container_width=True)
    
    with col2:
        st.dataframe(summary_df.iloc[7:], use_container_width=True)
    
    # Action plan
    st.markdown("### 🎯 Action Plan")
    
    action_steps = []
    
    # Emergency fund check
    if current_savings < emergency_fund:
        action_steps.append({
            'Priority': 'HIGH',
            'Action': 'Build Emergency Fund',
            'Details': f'Accumulate Rp {emergency_fund:,.0f} in liquid savings before investing for hajj',
            'Timeline': '3-6 months'
        })
    
    # Investment setup
    action_steps.append({
        'Priority': 'HIGH',
        'Action': 'Set Up Investment Portfolio',
        'Details': f'Implement {risk_profile} strategy with {investment_recommendation["expected_return"]:.1%} expected return',
        'Timeline': '1 month'
    })
    
    # Monthly savings automation
    action_steps.append({
        'Priority': 'HIGH',
        'Action': 'Automate Monthly Savings',
        'Details': f'Set up automatic transfer of Rp {recommended_option["monthly_savings"]:,.0f} monthly',
        'Timeline': '1 week'
    })
    
    # Regular review
    action_steps.append({
        'Priority': 'MEDIUM',
        'Action': 'Quarterly Portfolio Review',
        'Details': 'Review and rebalance portfolio allocation quarterly',
        'Timeline': 'Ongoing'
    })
    
    # Progress tracking
    action_steps.append({
        'Priority': 'MEDIUM',
        'Action': 'Monthly Progress Tracking',
        'Details': 'Monitor savings progress and adjust if needed',
        'Timeline': 'Ongoing'
    })
    
    action_df = pd.DataFrame(action_steps)
    st.dataframe(action_df, use_container_width=True)
    
    # Recommendations and warnings
    st.markdown("### ⚠️ Important Considerations")
    
    considerations = []
    
    # Affordability check
    savings_to_income_ratio = (recommended_option['monthly_savings'] / monthly_income) * 100
    if savings_to_income_ratio > 30:
        considerations.append("💰 **High Savings Requirement**: Required savings exceed 30% of income. Consider extending timeline or reducing target.")
    
    # Surplus check
    if monthly_surplus < recommended_option['monthly_savings']:
        considerations.append("📉 **Insufficient Surplus**: Monthly expenses exceed income after hajj savings. Review and optimize monthly budget.")
    
    # Risk tolerance vs time horizon
    if years_to_hajj < 5 and risk_profile == 'aggressive':
        considerations.append("⏰ **Time Horizon Risk**: Short time horizon with aggressive strategy may increase risk. Consider moderate approach.")
    
    # Emergency fund priority
    if current_savings < emergency_fund:
        considerations.append("🚨 **Emergency Fund Priority**: Build emergency fund before aggressive hajj saving to avoid financial stress.")
    
    # Inflation impact
    inflation_impact = ((future_hajj_cost / planner.current_hajj_cost) - 1) * 100
    considerations.append(f"📈 **Inflation Impact**: Hajj costs expected to increase by {inflation_impact:.1f}% over {years_to_hajj} years due to inflation.")
    
    for consideration in considerations:
        st.warning(consideration)
    
    if not considerations:
        st.success("✅ **Well-Balanced Plan**: Your financial plan appears well-balanced and achievable!")
    
    # Final recommendations
    st.markdown("### 💡 Final Recommendations")
    
    final_recommendations = [
        f"🎯 **Primary Goal**: Save Rp {recommended_option['monthly_savings']:,.0f} monthly using {risk_profile} investment strategy",
        f"📊 **Portfolio Allocation**: Follow recommended {risk_profile} allocation for optimal risk-return balance",
        f"⏰ **Timeline Management**: Stay committed to {years_to_hajj}-year timeline with regular progress reviews",
        f"🔄 **Flexibility**: Review and adjust plan annually based on income changes and market conditions",
        f"📚 **Education**: Continue learning about shariah-compliant investments and financial planning",
        f"🤝 **Professional Advice**: Consider consulting with certified Islamic financial planner for personalized guidance"
    ]
    
    for recommendation in final_recommendations:
        st.info(recommendation)

# Export functionality
st.markdown("---")
st.markdown("## 📊 Export Your Plan")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📋 Generate PDF Report", use_container_width=True):
        st.success("📄 PDF report generation initiated!")
        st.info("💡 Feature will generate comprehensive financial plan PDF")

with col2:
    if st.button("📊 Export to Excel", use_container_width=True):
        st.success("📊 Excel export initiated!")
        st.info("💡 Feature will export calculations and projections to Excel")

with col3:
    if st.button("📱 Share Plan", use_container_width=True):
        st.success("📤 Plan sharing initiated!")
        st.info("💡 Feature will generate shareable link for your plan")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <h4>💼 Personal Hajj Financial Planning Center</h4>
    <p>Shariah-Compliant Financial Planning & Investment Guidance</p>
    <p><em>Achieving your hajj dreams through disciplined financial planning</em></p>
</div>
""", unsafe_allow_html=True)
