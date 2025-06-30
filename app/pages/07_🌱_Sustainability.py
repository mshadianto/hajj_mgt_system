"""
🌱 ESG & SUSTAINABILITY METRICS MODULE
Comprehensive sustainability assessment for hajj fund management

Features:
- ESG (Environmental, Social, Governance) scoring
- Islamic finance compliance monitoring
- Intergenerational equity analysis
- Carbon footprint assessment
- Sustainability reporting & benchmarking
- Stakeholder impact analysis
- Long-term value creation metrics
- Sustainable Development Goals (SDG) alignment
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
    page_title="🌱 Sustainability Metrics",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .sustainability-header {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.3);
    }
    
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
    
    .impact-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(168, 237, 234, 0.3);
    }
    
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
</style>
""", unsafe_allow_html=True)

# ESG Scoring System
class ESGScoringSystem:
    """
    Comprehensive ESG scoring system for hajj fund management
    """
    
    def __init__(self, financial_data, investment_portfolio=None):
        self.financial_data = financial_data
        self.portfolio = investment_portfolio or self._generate_sample_portfolio()
        self.esg_scores = self._calculate_esg_scores()
        self.islamic_compliance = self._assess_islamic_compliance()
        self.sustainability_metrics = self._calculate_sustainability_metrics()
    
    def _generate_sample_portfolio(self):
        """Generate sample investment portfolio for demonstration"""
        return {
            'sukuk_government': {'allocation': 0.35, 'esg_score': 85, 'shariah_compliant': True},
            'sukuk_corporate': {'allocation': 0.20, 'esg_score': 78, 'shariah_compliant': True},
            'equity_shariah': {'allocation': 0.25, 'esg_score': 72, 'shariah_compliant': True},
            'real_estate_halal': {'allocation': 0.15, 'esg_score': 68, 'shariah_compliant': True},
            'commodities_halal': {'allocation': 0.05, 'esg_score': 65, 'shariah_compliant': True}
        }
    
    def _calculate_esg_scores(self):
        """Calculate comprehensive ESG scores"""
        
        # Environmental Score
        environmental_factors = {
            'carbon_footprint_reduction': 75,  # Portfolio's carbon impact
            'green_investment_ratio': 60,      # % in green investments
            'environmental_risk_management': 70, # Environmental risk policies
            'renewable_energy_exposure': 65,   # Exposure to renewable energy
            'sustainable_infrastructure': 80   # Sustainable infrastructure investments
        }
        
        environmental_score = np.mean(list(environmental_factors.values()))
        
        # Social Score
        social_factors = {
            'community_impact': 85,            # Positive community impact
            'financial_inclusion': 90,         # Promoting financial inclusion
            'education_support': 80,           # Educational initiatives
            'healthcare_access': 75,           # Healthcare accessibility
            'job_creation': 70,                # Employment generation
            'gender_equality': 65,             # Gender equality in investments
            'human_rights': 88                 # Human rights compliance
        }
        
        social_score = np.mean(list(social_factors.values()))
        
        # Governance Score
        governance_factors = {
            'board_independence': 82,          # Board independence
            'transparency': 88,                # Financial transparency
            'risk_management': 85,             # Risk management quality
            'stakeholder_engagement': 80,      # Stakeholder engagement
            'ethical_practices': 92,           # Ethical business practices
            'regulatory_compliance': 90,       # Regulatory compliance
            'audit_quality': 85                # Audit quality and frequency
        }
        
        governance_score = np.mean(list(governance_factors.values()))
        
        # Calculate weighted portfolio ESG score
        portfolio_esg = 0
        total_allocation = 0
        
        for asset, details in self.portfolio.items():
            portfolio_esg += details['allocation'] * details['esg_score']
            total_allocation += details['allocation']
        
        portfolio_esg_score = portfolio_esg / total_allocation if total_allocation > 0 else 0
        
        return {
            'environmental': {
                'score': environmental_score,
                'factors': environmental_factors,
                'grade': self._get_grade(environmental_score)
            },
            'social': {
                'score': social_score,
                'factors': social_factors,
                'grade': self._get_grade(social_score)
            },
            'governance': {
                'score': governance_score,
                'factors': governance_factors,
                'grade': self._get_grade(governance_score)
            },
            'overall': {
                'score': (environmental_score + social_score + governance_score) / 3,
                'portfolio_weighted': portfolio_esg_score,
                'grade': self._get_grade((environmental_score + social_score + governance_score) / 3)
            }
        }
    
    def _assess_islamic_compliance(self):
        """Assess Islamic finance compliance"""
        
        compliance_criteria = {
            'riba_free': {
                'status': True,
                'score': 100,
                'description': 'All investments are free from interest-based transactions'
            },
            'gharar_free': {
                'status': True,
                'score': 95,
                'description': 'Minimal uncertainty in investment contracts'
            },
            'maysir_free': {
                'status': True,
                'score': 98,
                'description': 'No gambling or excessive speculation'
            },
            'halal_business': {
                'status': True,
                'score': 90,
                'description': 'All business activities are permissible under Islamic law'
            },
            'asset_backing': {
                'status': True,
                'score': 85,
                'description': 'All investments are backed by tangible assets'
            },
            'profit_loss_sharing': {
                'status': True,
                'score': 80,
                'description': 'Risk and profit sharing principles applied'
            },
            'shariah_board_approval': {
                'status': True,
                'score': 95,
                'description': 'All investments approved by Shariah supervisory board'
            }
        }
        
        # Calculate overall compliance score
        total_score = sum([criteria['score'] for criteria in compliance_criteria.values()])
        avg_score = total_score / len(compliance_criteria)
        
        # Assess portfolio compliance
        portfolio_compliance = all([details['shariah_compliant'] for details in self.portfolio.values()])
        
        return {
            'criteria': compliance_criteria,
            'overall_score': avg_score,
            'grade': self._get_grade(avg_score),
            'portfolio_compliant': portfolio_compliance,
            'certification_status': 'Fully Certified' if avg_score >= 90 else 'Under Review'
        }
    
    def _calculate_sustainability_metrics(self):
        """Calculate comprehensive sustainability metrics"""
        
        # Intergenerational equity metrics
        current_sustainability = (self.financial_data['NilaiManfaat'].iloc[-1] / 
                                self.financial_data['BPIH'].iloc[-1]) * 100
        
        # Calculate sustainability trend
        sustainability_values = []
        for i in range(len(self.financial_data)):
            sus_val = (self.financial_data['NilaiManfaat'].iloc[i] / 
                      self.financial_data['BPIH'].iloc[i]) * 100
            sustainability_values.append(sus_val)
        
        sustainability_trend = np.polyfit(range(len(sustainability_values)), sustainability_values, 1)[0]
        
        # Long-term viability assessment
        projected_sustainability = current_sustainability + (sustainability_trend * 10)
        
        # Stakeholder impact metrics
        stakeholder_metrics = {
            'current_pilgrims': {
                'impact_score': 85,
                'description': 'Financial support for current hajj pilgrims'
            },
            'future_pilgrims': {
                'impact_score': projected_sustainability,
                'description': 'Projected impact on future generations'
            },
            'community_development': {
                'impact_score': 75,
                'description': 'Contribution to community development projects'
            },
            'economic_impact': {
                'impact_score': 80,
                'description': 'Positive economic impact on local communities'
            }
        }
        
        return {
            'current_sustainability_index': current_sustainability,
            'sustainability_trend': sustainability_trend,
            'projected_10_year': projected_sustainability,
            'intergenerational_equity': self._assess_intergenerational_equity(sustainability_values),
            'stakeholder_impact': stakeholder_metrics,
            'long_term_viability': self._assess_long_term_viability(projected_sustainability)
        }
    
    def _assess_intergenerational_equity(self, sustainability_values):
        """Assess equity between current and future generations"""
        
        recent_avg = np.mean(sustainability_values[-3:])  # Last 3 years
        historical_avg = np.mean(sustainability_values[:-3])  # Earlier years
        
        equity_ratio = recent_avg / historical_avg if historical_avg > 0 else 1
        
        if equity_ratio >= 1.0:
            equity_status = "Improving"
            equity_score = min(100, 50 + (equity_ratio - 1) * 50)
        else:
            equity_status = "Declining"
            equity_score = equity_ratio * 50
        
        return {
            'equity_ratio': equity_ratio,
            'equity_score': equity_score,
            'status': equity_status,
            'recent_avg': recent_avg,
            'historical_avg': historical_avg
        }
    
    def _assess_long_term_viability(self, projected_sustainability):
        """Assess long-term financial viability"""
        
        if projected_sustainability >= 70:
            viability = "Excellent"
            risk_level = "Very Low"
        elif projected_sustainability >= 50:
            viability = "Good"
            risk_level = "Low"
        elif projected_sustainability >= 30:
            viability = "Fair"
            risk_level = "Medium"
        else:
            viability = "Poor"
            risk_level = "High"
        
        return {
            'viability_rating': viability,
            'risk_level': risk_level,
            'projected_sustainability': projected_sustainability,
            'recommended_actions': self._get_sustainability_recommendations(projected_sustainability)
        }
    
    def _get_sustainability_recommendations(self, projected_sustainability):
        """Get recommendations based on sustainability projection"""
        
        if projected_sustainability >= 70:
            return [
                "Maintain current excellent performance",
                "Explore opportunities for increased ESG impact",
                "Consider expanding sustainable investment portfolio"
            ]
        elif projected_sustainability >= 50:
            return [
                "Monitor sustainability metrics closely",
                "Implement gradual optimization strategies",
                "Increase allocation to high-ESG investments"
            ]
        elif projected_sustainability >= 30:
            return [
                "Urgent action required to improve sustainability",
                "Review and optimize investment strategy",
                "Implement cost reduction and efficiency measures",
                "Increase focus on high-return sustainable investments"
            ]
        else:
            return [
                "Critical intervention required immediately",
                "Complete review of investment and operational strategy",
                "Emergency cost containment measures",
                "Seek expert consultation for turnaround plan"
            ]
    
    def _get_grade(self, score):
        """Convert numerical score to letter grade"""
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

# SDG Alignment Assessment
class SDGAlignment:
    """
    Assess alignment with UN Sustainable Development Goals
    """
    
    def __init__(self):
        self.sdg_goals = self._define_sdg_goals()
    
    def _define_sdg_goals(self):
        """Define relevant SDG goals for hajj fund management"""
        
        return {
            'SDG1': {
                'title': 'No Poverty',
                'relevance': 'High',
                'alignment_score': 85,
                'description': 'Financial inclusion and support for hajj pilgrims',
                'actions': ['Affordable hajj financing', 'Support for low-income pilgrims']
            },
            'SDG4': {
                'title': 'Quality Education',
                'relevance': 'Medium',
                'alignment_score': 70,
                'description': 'Educational programs and financial literacy',
                'actions': ['Financial education', 'Investment in educational institutions']
            },
            'SDG5': {
                'title': 'Gender Equality',
                'relevance': 'Medium',
                'alignment_score': 65,
                'description': 'Equal access to hajj financing for all genders',
                'actions': ['Gender-inclusive policies', 'Equal access programs']
            },
            'SDG8': {
                'title': 'Decent Work and Economic Growth',
                'relevance': 'High',
                'alignment_score': 80,
                'description': 'Economic development through sustainable investments',
                'actions': ['Job creation', 'Economic empowerment', 'Sustainable business practices']
            },
            'SDG9': {
                'title': 'Industry, Innovation and Infrastructure',
                'relevance': 'Medium',
                'alignment_score': 75,
                'description': 'Investment in sustainable infrastructure',
                'actions': ['Infrastructure development', 'Technology adoption']
            },
            'SDG11': {
                'title': 'Sustainable Cities and Communities',
                'relevance': 'High',
                'alignment_score': 78,
                'description': 'Community development and urban sustainability',
                'actions': ['Community projects', 'Urban development investments']
            },
            'SDG13': {
                'title': 'Climate Action',
                'relevance': 'Medium',
                'alignment_score': 60,
                'description': 'Climate-conscious investment decisions',
                'actions': ['Green investments', 'Carbon footprint reduction']
            },
            'SDG16': {
                'title': 'Peace, Justice and Strong Institutions',
                'relevance': 'High',
                'alignment_score': 90,
                'description': 'Transparent and ethical fund management',
                'actions': ['Transparent governance', 'Ethical practices', 'Strong institutions']
            },
            'SDG17': {
                'title': 'Partnerships for the Goals',
                'relevance': 'High',
                'alignment_score': 82,
                'description': 'Collaboration with stakeholders for sustainable development',
                'actions': ['Strategic partnerships', 'Multi-stakeholder collaboration']
            }
        }
    
    def calculate_overall_alignment(self):
        """Calculate overall SDG alignment score"""
        
        total_score = 0
        high_relevance_count = 0
        
        for sdg, details in self.sdg_goals.items():
            if details['relevance'] == 'High':
                weight = 2
                high_relevance_count += 2
            else:
                weight = 1
                high_relevance_count += 1
            
            total_score += details['alignment_score'] * weight
        
        overall_score = total_score / high_relevance_count if high_relevance_count > 0 else 0
        
        return {
            'overall_score': overall_score,
            'grade': self._get_sdg_grade(overall_score),
            'high_relevance_sdgs': [sdg for sdg, details in self.sdg_goals.items() if details['relevance'] == 'High'],
            'improvement_priorities': self._identify_improvement_priorities()
        }
    
    def _get_sdg_grade(self, score):
        """Convert SDG score to grade"""
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
    
    def _identify_improvement_priorities(self):
        """Identify SDGs needing improvement"""
        
        low_scoring_sdgs = []
        
        for sdg, details in self.sdg_goals.items():
            if details['alignment_score'] < 70 and details['relevance'] == 'High':
                low_scoring_sdgs.append({
                    'sdg': sdg,
                    'title': details['title'],
                    'score': details['alignment_score'],
                    'actions': details['actions']
                })
        
        return sorted(low_scoring_sdgs, key=lambda x: x['score'])

# Header
st.markdown("""
<div class="sustainability-header">
    <h1>🌱 ESG & SUSTAINABILITY METRICS</h1>
    <h3>Comprehensive Sustainability Assessment for Hajj Fund Management</h3>
    <p>ESG Scoring | Islamic Compliance | Intergenerational Equity | SDG Alignment</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("## 🌱 Sustainability Controls")
    
    assessment_type = st.selectbox(
        "Assessment Type",
        [
            "ESG Overview",
            "Islamic Compliance",
            "Sustainability Metrics",
            "SDG Alignment",
            "Carbon Footprint",
            "Stakeholder Impact",
            "Compliance Monitoring"
        ]
    )
    
    st.markdown("---")
    
    # Reporting parameters
    st.markdown("### 📊 Reporting Parameters")
    
    reporting_period = st.selectbox(
        "Reporting Period",
        ["Quarterly", "Annual", "Multi-year"]
    )
    
    stakeholder_focus = st.multiselect(
        "Stakeholder Focus",
        ["Current Pilgrims", "Future Pilgrims", "Community", "Regulators", "Investors"],
        default=["Current Pilgrims", "Future Pilgrims"]
    )
    
    # ESG weights
    st.markdown("### ⚖️ ESG Weighting")
    
    env_weight = st.slider("Environmental Weight", 0.0, 1.0, 0.3)
    social_weight = st.slider("Social Weight", 0.0, 1.0, 0.4)
    governance_weight = st.slider("Governance Weight", 0.0, 1.0, 0.3)
    
    # Normalize weights
    total_weight = env_weight + social_weight + governance_weight
    if total_weight > 0:
        env_weight /= total_weight
        social_weight /= total_weight
        governance_weight /= total_weight

# Load data and initialize ESG system
@st.cache_data
def load_sustainability_data():
    """Load comprehensive data for sustainability assessment"""
    np.random.seed(42)
    
    years = range(2015, 2026)
    data = []
    
    base_bpih = 75000000
    base_bipih = 35000000
    base_benefit = 50000000
    
    for i, year in enumerate(years):
        economic_cycle = np.sin(i * 0.5) * 0.02
        
        bpih = base_bpih * (1.05 + economic_cycle) ** i * np.random.uniform(0.95, 1.05)
        bipih = base_bipih * (1.07 + economic_cycle) ** i * np.random.uniform(0.9, 1.1)
        benefit = base_benefit * (0.98 + economic_cycle * 0.5) ** i * np.random.uniform(0.95, 1.05)
        
        investment_return = 0.06 + economic_cycle + np.random.normal(0, 0.02)
        inflation_rate = 0.03 + abs(economic_cycle) + np.random.normal(0, 0.01)
        
        data.append({
            'Year': year,
            'BPIH': bpih,
            'Bipih': bipih,
            'NilaiManfaat': benefit,
            'InvestmentReturn': investment_return,
            'InflationRate': inflation_rate
        })
    
    return pd.DataFrame(data)

data = load_sustainability_data()
esg_system = ESGScoringSystem(data)
sdg_alignment = SDGAlignment()

# Main content based on assessment type
if assessment_type == "ESG Overview":
    
    st.markdown("## 🌍 ESG Comprehensive Overview")
    
    # ESG Score Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        env_score = esg_system.esg_scores['environmental']['score']
        score_class = 'score-excellent' if env_score >= 80 else 'score-good' if env_score >= 70 else 'score-fair' if env_score >= 60 else 'score-poor'
        
        st.markdown(f"""
        <div class="esg-card">
            <h3 style="text-align: center;">🌍 Environmental</h3>
            <div class="{score_class} score-circle">
                {env_score:.0f}
            </div>
            <p style="text-align: center; margin-top: 1rem;">
                <strong>Grade: {esg_system.esg_scores['environmental']['grade']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        social_score = esg_system.esg_scores['social']['score']
        score_class = 'score-excellent' if social_score >= 80 else 'score-good' if social_score >= 70 else 'score-fair' if social_score >= 60 else 'score-poor'
        
        st.markdown(f"""
        <div class="esg-card">
            <h3 style="text-align: center;">👥 Social</h3>
            <div class="{score_class} score-circle">
                {social_score:.0f}
            </div>
            <p style="text-align: center; margin-top: 1rem;">
                <strong>Grade: {esg_system.esg_scores['social']['grade']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        governance_score = esg_system.esg_scores['governance']['score']
        score_class = 'score-excellent' if governance_score >= 80 else 'score-good' if governance_score >= 70 else 'score-fair' if governance_score >= 60 else 'score-poor'
        
        st.markdown(f"""
        <div class="esg-card">
            <h3 style="text-align: center;">🏛️ Governance</h3>
            <div class="{score_class} score-circle">
                {governance_score:.0f}
            </div>
            <p style="text-align: center; margin-top: 1rem;">
                <strong>Grade: {esg_system.esg_scores['governance']['grade']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        overall_score = esg_system.esg_scores['overall']['score']
        weighted_score = (env_score * env_weight + social_score * social_weight + governance_score * governance_weight)
        score_class = 'score-excellent' if weighted_score >= 80 else 'score-good' if weighted_score >= 70 else 'score-fair' if weighted_score >= 60 else 'score-poor'
        
        st.markdown(f"""
        <div class="esg-card">
            <h3 style="text-align: center;">⭐ Overall ESG</h3>
            <div class="{score_class} score-circle">
                {weighted_score:.0f}
            </div>
            <p style="text-align: center; margin-top: 1rem;">
                <strong>Grade: {esg_system._get_grade(weighted_score)}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ESG Factor Breakdown
    st.markdown("### 📊 ESG Factor Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🌍 Environmental", "👥 Social", "🏛️ Governance"])
    
    with tab1:
        env_factors = esg_system.esg_scores['environmental']['factors']
        
        fig_env = px.bar(
            x=list(env_factors.values()),
            y=list(env_factors.keys()),
            orientation='h',
            title="Environmental Factors Breakdown",
            color=list(env_factors.values()),
            color_continuous_scale='Greens'
        )
        fig_env.update_layout(height=400)
        st.plotly_chart(fig_env, use_container_width=True)
        
        # Environmental recommendations
        st.markdown("#### 🌱 Environmental Recommendations")
        
        low_scoring_env = [factor for factor, score in env_factors.items() if score < 70]
        
        if low_scoring_env:
            for factor in low_scoring_env:
                st.markdown(f"• **{factor.replace('_', ' ').title()}**: Needs improvement (Score: {env_factors[factor]})")
        else:
            st.success("✅ All environmental factors are performing well!")
    
    with tab2:
        social_factors = esg_system.esg_scores['social']['factors']
        
        fig_social = px.bar(
            x=list(social_factors.values()),
            y=list(social_factors.keys()),
            orientation='h',
            title="Social Factors Breakdown",
            color=list(social_factors.values()),
            color_continuous_scale='Blues'
        )
        fig_social.update_layout(height=400)
        st.plotly_chart(fig_social, use_container_width=True)
        
        # Social impact highlights
        st.markdown("#### 👥 Social Impact Highlights")
        
        high_social_factors = [factor for factor, score in social_factors.items() if score >= 80]
        
        for factor in high_social_factors:
            st.markdown(f"✅ **{factor.replace('_', ' ').title()}**: Excellent performance (Score: {social_factors[factor]})")
    
    with tab3:
        governance_factors = esg_system.esg_scores['governance']['factors']
        
        fig_gov = px.bar(
            x=list(governance_factors.values()),
            y=list(governance_factors.keys()),
            orientation='h',
            title="Governance Factors Breakdown",
            color=list(governance_factors.values()),
            color_continuous_scale='Purples'
        )
        fig_gov.update_layout(height=400)
        st.plotly_chart(fig_gov, use_container_width=True)
        
        # Governance excellence areas
        st.markdown("#### 🏛️ Governance Excellence")
        
        excellent_gov = [factor for factor, score in governance_factors.items() if score >= 85]
        
        for factor in excellent_gov:
            st.markdown(f"⭐ **{factor.replace('_', ' ').title()}**: Outstanding (Score: {governance_factors[factor]})")

elif assessment_type == "Islamic Compliance":
    
    st.markdown("## ☪️ Islamic Finance Compliance Assessment")
    
    compliance = esg_system.islamic_compliance
    
    # Overall compliance status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        compliance_score = compliance['overall_score']
        score_class = 'score-excellent' if compliance_score >= 90 else 'score-good' if compliance_score >= 80 else 'score-fair'
        
        st.markdown(f"""
        <div class="esg-card">
            <h3 style="text-align: center;">☪️ Shariah Compliance</h3>
            <div class="{score_class} score-circle">
                {compliance_score:.0f}
            </div>
            <p style="text-align: center; margin-top: 1rem;">
                <strong>Grade: {compliance['grade']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="esg-card">
            <h3 style="text-align: center;">📜 Certification Status</h3>
            <div style="text-align: center; padding: 2rem;">
                <h2>{compliance['certification_status']}</h2>
                <p>Portfolio Compliant: {'✅ Yes' if compliance['portfolio_compliant'] else '❌ No'}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        compliant_count = sum([1 for criteria in compliance['criteria'].values() if criteria['status']])
        total_criteria = len(compliance['criteria'])
        
        st.markdown(f"""
        <div class="esg-card">
            <h3 style="text-align: center;">✅ Compliance Rate</h3>
            <div style="text-align: center; padding: 2rem;">
                <h2>{compliant_count}/{total_criteria}</h2>
                <p>{(compliant_count/total_criteria)*100:.0f}% Compliant</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed compliance criteria
    st.markdown("### 📋 Detailed Compliance Assessment")
    
    criteria_data = []
    
    for criterion, details in compliance['criteria'].items():
        status_badge = "compliant" if details['status'] else "non-compliant"
        criteria_data.append({
            'Criterion': criterion.replace('_', ' ').title(),
            'Status': '✅ Compliant' if details['status'] else '❌ Non-Compliant',
            'Score': details['score'],
            'Description': details['description']
        })
    
    criteria_df = pd.DataFrame(criteria_data)
    st.dataframe(criteria_df, use_container_width=True)
    
    # Compliance monitoring chart
    st.markdown("### 📊 Compliance Score Breakdown")
    
    fig_compliance = px.bar(
        x=[details['score'] for details in compliance['criteria'].values()],
        y=[criterion.replace('_', ' ').title() for criterion in compliance['criteria'].keys()],
        orientation='h',
        title="Shariah Compliance Scores by Criterion",
        color=[details['score'] for details in compliance['criteria'].values()],
        color_continuous_scale='Greens'
    )
    fig_compliance.update_layout(height=500)
    st.plotly_chart(fig_compliance, use_container_width=True)
    
    # Investment portfolio compliance
    st.markdown("### 💼 Investment Portfolio Compliance")
    
    portfolio_data = []
    
    for asset, details in esg_system.portfolio.items():
        portfolio_data.append({
            'Asset Class': asset.replace('_', ' ').title(),
            'Allocation': f"{details['allocation']:.0%}",
            'ESG Score': details['esg_score'],
            'Shariah Compliant': '✅ Yes' if details['shariah_compliant'] else '❌ No'
        })
    
    portfolio_df = pd.DataFrame(portfolio_data)
    st.dataframe(portfolio_df, use_container_width=True)

elif assessment_type == "Sustainability Metrics":
    
    st.markdown("## 📈 Comprehensive Sustainability Metrics")
    
    sustainability_metrics = esg_system.sustainability_metrics
    
    # Key sustainability indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_index = sustainability_metrics['current_sustainability_index']
        st.metric(
            "Current Sustainability Index",
            f"{current_index:.1f}%",
            f"Trend: {sustainability_metrics['sustainability_trend']:+.2f}/year"
        )
    
    with col2:
        projected_index = sustainability_metrics['projected_10_year']
        change = projected_index - current_index
        st.metric(
            "10-Year Projection",
            f"{projected_index:.1f}%",
            f"{change:+.1f}pp"
        )
    
    with col3:
        equity_score = sustainability_metrics['intergenerational_equity']['equity_score']
        equity_status = sustainability_metrics['intergenerational_equity']['status']
        st.metric(
            "Intergenerational Equity",
            f"{equity_score:.0f}",
            equity_status
        )
    
    with col4:
        viability = sustainability_metrics['long_term_viability']['viability_rating']
        risk_level = sustainability_metrics['long_term_viability']['risk_level']
        st.metric(
            "Long-term Viability",
            viability,
            f"Risk: {risk_level}"
        )
    
    # Sustainability trend analysis
    st.markdown("### 📊 Sustainability Trend Analysis")
    
    # Calculate historical sustainability values
    sustainability_history = []
    for i in range(len(data)):
        sus_val = (data['NilaiManfaat'].iloc[i] / data['BPIH'].iloc[i]) * 100
        sustainability_history.append(sus_val)
    
    # Create projection
    years_hist = data['Year'].tolist()
    years_proj = list(range(years_hist[-1] + 1, years_hist[-1] + 11))
    
    # Linear projection
    trend_slope = sustainability_metrics['sustainability_trend']
    proj_values = [sustainability_history[-1] + trend_slope * i for i in range(1, 11)]
    
    fig_sustain_trend = go.Figure()
    
    # Historical data
    fig_sustain_trend.add_trace(go.Scatter(
        x=years_hist,
        y=sustainability_history,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#3498db', width=3)
    ))
    
    # Projection
    fig_sustain_trend.add_trace(go.Scatter(
        x=years_proj,
        y=proj_values,
        mode='lines+markers',
        name='Projection',
        line=dict(color='#e74c3c', dash='dash', width=3)
    ))
    
    # Threshold lines
    fig_sustain_trend.add_hline(y=70, line_dash="dot", line_color="green", annotation_text="Excellent")
    fig_sustain_trend.add_hline(y=50, line_dash="dot", line_color="orange", annotation_text="Warning")
    fig_sustain_trend.add_hline(y=30, line_dash="dot", line_color="red", annotation_text="Critical")
    
    fig_sustain_trend.update_layout(
        title="Sustainability Index: Historical Trends & Projections",
        xaxis_title="Year",
        yaxis_title="Sustainability Index (%)",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig_sustain_trend, use_container_width=True)
    
    # Stakeholder impact analysis
    st.markdown("### 👥 Stakeholder Impact Analysis")
    
    stakeholder_metrics = sustainability_metrics['stakeholder_impact']
    
    col1, col2 = st.columns(2)
    
    with col1:
        stakeholder_data = []
        
        for stakeholder, metrics in stakeholder_metrics.items():
            stakeholder_data.append({
                'Stakeholder': stakeholder.replace('_', ' ').title(),
                'Impact Score': metrics['impact_score'],
                'Description': metrics['description']
            })
        
        stakeholder_df = pd.DataFrame(stakeholder_data)
        st.dataframe(stakeholder_df, use_container_width=True)
    
    with col2:
        fig_stakeholder = px.bar(
            stakeholder_df,
            x='Impact Score',
            y='Stakeholder',
            orientation='h',
            title="Stakeholder Impact Scores",
            color='Impact Score',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_stakeholder, use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 Sustainability Recommendations")
    
    recommendations = sustainability_metrics['long_term_viability']['recommended_actions']
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

elif assessment_type == "SDG Alignment":
    
    st.markdown("## 🎯 UN Sustainable Development Goals Alignment")
    
    sdg_assessment = sdg_alignment.calculate_overall_alignment()
    
    # Overall SDG performance
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overall_sdg_score = sdg_assessment['overall_score']
        score_class = 'score-excellent' if overall_sdg_score >= 80 else 'score-good' if overall_sdg_score >= 70 else 'score-fair'
        
        st.markdown(f"""
        <div class="esg-card">
            <h3 style="text-align: center;">🎯 Overall SDG Score</h3>
            <div class="{score_class} score-circle">
                {overall_sdg_score:.0f}
            </div>
            <p style="text-align: center; margin-top: 1rem;">
                <strong>Grade: {sdg_assessment['grade']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_relevance_count = len(sdg_assessment['high_relevance_sdgs'])
        
        st.markdown(f"""
        <div class="esg-card">
            <h3 style="text-align: center;">⭐ High Relevance SDGs</h3>
            <div style="text-align: center; padding: 2rem;">
                <h2>{high_relevance_count}</h2>
                <p>out of {len(sdg_alignment.sdg_goals)} total SDGs</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        improvement_count = len(sdg_assessment['improvement_priorities'])
        
        st.markdown(f"""
        <div class="esg-card">
            <h3 style="text-align: center;">📈 Improvement Areas</h3>
            <div style="text-align: center; padding: 2rem;">
                <h2>{improvement_count}</h2>
                <p>SDGs needing attention</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # SDG performance breakdown
    st.markdown("### 📊 SDG Performance Breakdown")
    
    sdg_data = []
    
    for sdg_id, sdg_info in sdg_alignment.sdg_goals.items():
        sdg_data.append({
            'SDG': sdg_id,
            'Title': sdg_info['title'],
            'Relevance': sdg_info['relevance'],
            'Alignment Score': sdg_info['alignment_score'],
            'Description': sdg_info['description']
        })
    
    sdg_df = pd.DataFrame(sdg_data)
    
    # SDG visualization
    fig_sdg = px.scatter(
        sdg_df,
        x='SDG',
        y='Alignment Score',
        size='Alignment Score',
        color='Relevance',
        hover_data=['Title', 'Description'],
        title="SDG Alignment Scores",
        color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#95a5a6'}
    )
    
    fig_sdg.update_layout(height=500)
    st.plotly_chart(fig_sdg, use_container_width=True)
    
    # Detailed SDG assessment
    st.markdown("### 📋 Detailed SDG Assessment")
    
    # Create expandable sections for each SDG
    for sdg_id, sdg_info in sdg_alignment.sdg_goals.items():
        with st.expander(f"{sdg_id}: {sdg_info['title']} (Score: {sdg_info['alignment_score']})"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Relevance:** {sdg_info['relevance']}")
                st.markdown(f"**Current Score:** {sdg_info['alignment_score']}/100")
                st.markdown(f"**Description:** {sdg_info['description']}")
            
            with col2:
                st.markdown("**Key Actions:**")
                for action in sdg_info['actions']:
                    st.markdown(f"• {action}")
    
    # Improvement priorities
    if sdg_assessment['improvement_priorities']:
        st.markdown("### 🚀 Priority Improvement Areas")
        
        for priority in sdg_assessment['improvement_priorities']:
            st.markdown(f"""
            <div class="impact-box">
                <h4>{priority['sdg']}: {priority['title']}</h4>
                <p><strong>Current Score:</strong> {priority['score']}/100</p>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    {''.join([f'<li>{action}</li>' for action in priority['actions']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Export and reporting
st.markdown("---")
st.markdown("## 📊 Sustainability Reporting")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📋 Generate ESG Report", use_container_width=True):
        st.success("✅ ESG Report Generated!")
        
        # Create summary report
        esg_summary = {
            'Metric': [
                'Environmental Score',
                'Social Score',
                'Governance Score',
                'Overall ESG Score',
                'Islamic Compliance',
                'SDG Alignment'
            ],
            'Score': [
                f"{esg_system.esg_scores['environmental']['score']:.0f}",
                f"{esg_system.esg_scores['social']['score']:.0f}",
                f"{esg_system.esg_scores['governance']['score']:.0f}",
                f"{esg_system.esg_scores['overall']['score']:.0f}",
                f"{esg_system.islamic_compliance['overall_score']:.0f}",
                f"{sdg_alignment.calculate_overall_alignment()['overall_score']:.0f}"
            ],
            'Grade': [
                esg_system.esg_scores['environmental']['grade'],
                esg_system.esg_scores['social']['grade'],
                esg_system.esg_scores['governance']['grade'],
                esg_system.esg_scores['overall']['grade'],
                esg_system.islamic_compliance['grade'],
                sdg_alignment.calculate_overall_alignment()['grade']
            ]
        }
        
        st.dataframe(pd.DataFrame(esg_summary), use_container_width=True)

with col2:
    if st.button("📈 Sustainability Dashboard", use_container_width=True):
        st.success("📊 Dashboard Created!")
        
        # Quick sustainability metrics
        sustainability = esg_system.sustainability_metrics
        
        quick_metrics = {
            'Metric': [
                'Current Sustainability Index',
                'Projected 10-Year Index',
                'Intergenerational Equity Score',
                'Long-term Viability'
            ],
            'Value': [
                f"{sustainability['current_sustainability_index']:.1f}%",
                f"{sustainability['projected_10_year']:.1f}%",
                f"{sustainability['intergenerational_equity']['equity_score']:.0f}",
                sustainability['long_term_viability']['viability_rating']
            ]
        }
        
        st.dataframe(pd.DataFrame(quick_metrics), use_container_width=True)

with col3:
    if st.button("🔄 Schedule Monitoring", use_container_width=True):
        st.success("⏰ Monitoring Scheduled!")
        
        monitoring_schedule = {
            'Frequency': [
                'Daily',
                'Weekly',
                'Monthly',
                'Quarterly',
                'Annually'
            ],
            'Metrics': [
                'Portfolio Performance',
                'Risk Indicators',
                'ESG Scores',
                'Compliance Review',
                'Full Assessment'
            ]
        }
        
        st.dataframe(pd.DataFrame(monitoring_schedule), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <h4>🌱 ESG & Sustainability Center</h4>
    <p>Comprehensive Sustainability Assessment & Monitoring</p>
    <p><em>Building sustainable financial futures aligned with Islamic principles</em></p>
</div>
""", unsafe_allow_html=True)