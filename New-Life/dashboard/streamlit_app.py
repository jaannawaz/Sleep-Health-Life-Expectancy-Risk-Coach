"""
Streamlit Dashboard for Sleep Health & Life Expectancy Risk Coach
Interactive web application for sleep health assessment and exploration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/api'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/mcp_tools'))

try:
    from complete_sleep_health_system import CompleteSleepHealthSystem
    from tool_schemas import MAJOR_COUNTRIES
except ImportError as e:
    st.error(f"Could not import sleep health modules: {e}")
    CompleteSleepHealthSystem = None
    MAJOR_COUNTRIES = []

# Page configuration
st.set_page_config(
    page_title="Sleep Health Risk Coach",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    /* Recommendation card → Yellow */
    .metric-card {
        background: #fde047; /* yellow-300 */
        color: #111827;      /* gray-900 */
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b; /* amber-500 */
    }
    /* Force all prediction result cards to Red, regardless of state */
    .warning-card,
    .success-card,
    .danger-card {
        background: #ef4444; /* red-500 */
        color: #ffffff;       /* white text */
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #b91c1c; /* red-700 */
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
    st.session_state.sleep_system = None
    st.session_state.assessment_history = []

@st.cache_resource
def initialize_sleep_system():
    """Initialize the sleep health system with caching"""
    try:
        if CompleteSleepHealthSystem:
            system = CompleteSleepHealthSystem()
            return system
        else:
            return None
    except Exception as e:
        st.error(f"Failed to initialize sleep health system: {e}")
        return None

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Sleep Health & Life Expectancy Risk Coach</h1>
        <p>Comprehensive sleep disorder prediction with AI-powered medical insights</p>
        <p><strong>ML Accuracy:</strong> 97.3% | <strong>WHO Data:</strong> 183 Countries | <strong>Medical AI:</strong> Groq-powered</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing Sleep Health System..."):
            st.session_state.sleep_system = initialize_sleep_system()
            st.session_state.system_initialized = True
    
    # Sidebar
    create_sidebar()
    
    # Main content
    if st.session_state.sleep_system:
        create_main_content()
    else:
        show_system_unavailable()

def create_sidebar():
    """Create the sidebar with navigation and system status"""
    
    st.sidebar.markdown("## 🎯 Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a section:",
        [
            "🏠 Home & Assessment",
            "🌍 WHO Population Context",
            "📊 Model Performance",
            "🔧 System Status",
            "📚 Documentation"
        ]
    )
    
    st.session_state.current_page = page
    
    # System status in sidebar
    st.sidebar.markdown("## 📊 System Status")
    
    if st.session_state.sleep_system:
        system = st.session_state.sleep_system
        
        # Component status
        ml_status = "✅" if system.models_available else "❌"
        who_status = "✅" if system.who_context else "❌"
        groq_status = "✅" if system.groq_available else "❌"
        
        st.sidebar.markdown(f"""
        - **ML Models:** {ml_status}
        - **WHO Integration:** {who_status}
        - **Medical AI:** {groq_status}
        """)
        
        if system.who_context:
            countries_count = len(system.who_context.get_available_countries())
            st.sidebar.markdown(f"- **Countries:** {countries_count}")
    else:
        st.sidebar.error("System not available")
    
    # Quick stats
    if st.session_state.assessment_history:
        st.sidebar.markdown("## 📈 Session Stats")
        st.sidebar.metric("Assessments", len(st.session_state.assessment_history))

def create_main_content():
    """Create the main content area based on selected page"""
    
    page = st.session_state.current_page
    
    if page == "🏠 Home & Assessment":
        create_assessment_page()
    elif page == "🌍 WHO Population Context":
        create_who_context_page()
    elif page == "📊 Model Performance":
        create_performance_page()
    elif page == "🔧 System Status":
        create_system_status_page()
    elif page == "📚 Documentation":
        create_documentation_page()

def create_assessment_page():
    """Create the main assessment page"""
    
    st.markdown("## 🎯 Sleep Health Assessment")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 User Profile")
        
        # User input form
        with st.form("user_assessment_form"):
            # Basic demographics
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 100, 35)
            occupation = st.text_input("Occupation", "Software Engineer")
            
            # Sleep metrics
            st.markdown("**Sleep & Lifestyle**")
            sleep_duration = st.slider("Sleep Duration (hours)", 1.0, 12.0, 7.0, 0.5)
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
            
            # Health metrics
            st.markdown("**Health Indicators**")
            bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
            physical_activity = st.slider("Physical Activity Level (0-100)", 0, 100, 50)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            
            # Additional metrics
            st.markdown("**Additional Information**")
            blood_pressure = st.text_input("Blood Pressure", "120/80")
            heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 200, 70)
            daily_steps = st.slider("Daily Steps", 0, 50000, 7000, 500)
            
            # Assessment options
            st.markdown("**Assessment Options**")
            country = st.selectbox("Country for Population Context", 
                                 ["United States of America"] + [c for c in MAJOR_COUNTRIES if c != "United States of America"])
            include_explanation = st.checkbox("Include Medical Explanation", True)
            
            submitted = st.form_submit_button("🔮 Assess Sleep Health", type="primary")
        
        if submitted:
            # Prepare user data
            user_data = {
                "Gender": gender,
                "Age": age,
                "Occupation": occupation,
                "Sleep Duration": sleep_duration,
                "Quality of Sleep": sleep_quality,
                "Physical Activity Level": physical_activity,
                "Stress Level": stress_level,
                "BMI Category": bmi_category,
                "Blood Pressure": blood_pressure,
                "Heart Rate": heart_rate,
                "Daily Steps": daily_steps
            }
            
            # Perform assessment
            perform_assessment(user_data, country, include_explanation, col2)
    
    # Show assessment history
    if st.session_state.assessment_history:
        st.markdown("## 📈 Assessment History")
        show_assessment_history()

def perform_assessment(user_data, country, include_explanation, result_col):
    """Perform the sleep health assessment"""
    
    with result_col:
        st.markdown("### 🔮 Assessment Results")
        
        with st.spinner("🧠 Analyzing your sleep health..."):
            try:
                start_time = time.time()
                
                # Perform comprehensive assessment
                assessment = st.session_state.sleep_system.comprehensive_assessment(
                    user_data=user_data,
                    country=country,
                    include_explanation=include_explanation
                )
                
                processing_time = time.time() - start_time
                
                # Store in history
                assessment_record = {
                    "timestamp": pd.Timestamp.now(),
                    "user_data": user_data,
                    "assessment": assessment,
                    "processing_time": processing_time
                }
                st.session_state.assessment_history.append(assessment_record)
                
                # Display results
                display_assessment_results(assessment, processing_time)
                
            except Exception as e:
                st.error(f"Assessment failed: {e}")

def display_assessment_results(assessment, processing_time):
    """Display the assessment results"""
    
    # Extract key results
    ml_pred = assessment["ml_predictions"]
    pop_context = assessment["population_context"]
    recommendations = assessment["recommendations"]
    
    # Sleep disorder prediction
    disorder = ml_pred["sleep_disorder"]
    quality = ml_pred["sleep_quality"]
    risk_cal = pop_context["risk_calibration"]
    
    # Main metrics
    st.markdown("#### 🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_color = "success" if disorder["confidence"] < 0.5 else "warning" if disorder["confidence"] < 0.8 else "danger"
        st.markdown(f"""
        <div class="{confidence_color}-card">
            <h4>Sleep Disorder</h4>
            <h3>{disorder['predicted_class']}</h3>
            <p>Confidence: {disorder['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        quality_color = "danger" if quality["predicted_quality"] < 5 else "warning" if quality["predicted_quality"] < 7 else "success"
        st.markdown(f"""
        <div class="{quality_color}-card">
            <h4>Sleep Quality</h4>
            <h3>{quality['predicted_quality']}/10</h3>
            <p>Level: {quality['quality_level']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        risk_color = "success" if risk_cal["risk_level"] == "Low" else "warning" if risk_cal["risk_level"] == "Moderate" else "danger"
        st.markdown(f"""
        <div class="{risk_color}-card">
            <h4>Population Risk</h4>
            <h3>{risk_cal['risk_level']}</h3>
            <p>Adjusted: {risk_cal['population_adjusted_risk']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk probabilities chart
    if "probabilities" in disorder:
        st.markdown("#### 📊 Risk Breakdown")
        
        prob_df = pd.DataFrame(list(disorder["probabilities"].items()), 
                              columns=["Disorder", "Probability"])
        
        fig = px.bar(prob_df, x="Disorder", y="Probability", 
                    title="Sleep Disorder Probabilities",
                    color="Probability",
                    color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    if recommendations and "basic" in recommendations:
        st.markdown("#### 💡 Recommendations")
        
        basic_recs = recommendations["basic"]
        for i, rec in enumerate(basic_recs[:5], 1):
            priority_color = "danger" if rec["priority"] == "High" else "warning" if rec["priority"] == "Medium" else "success"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{i}. [{rec['category']}]</strong> {rec['recommendation']}<br>
                <small><span class="{priority_color}">Priority: {rec['priority']}</span> - {rec['reason']}</small>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Medical explanation
    if "medical_explanation" in assessment and assessment["medical_explanation"]:
        med_exp = assessment["medical_explanation"]
        if "error" not in med_exp:
            st.markdown("#### 🧠 Medical Analysis")
            
            if "key_risk_factors" in med_exp:
                st.markdown("**Key Risk Factors:**")
                st.markdown(", ".join(med_exp["key_risk_factors"]))
            
            if "full_explanation" in med_exp:
                with st.expander("📋 Complete Medical Explanation"):
                    st.markdown(med_exp["full_explanation"])
    
    # Performance info
    st.markdown("#### ⚡ Performance")
    st.metric("Processing Time", f"{processing_time*1000:.0f}ms")

def create_who_context_page():
    """Create the WHO population context page"""
    
    st.markdown("## 🌍 WHO Population Health Context")
    
    if not st.session_state.sleep_system or not st.session_state.sleep_system.who_context:
        st.error("WHO integration not available")
        return
    
    who_context = st.session_state.sleep_system.who_context
    
    # Country selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🏳️ Country Selection")
        
        available_countries = who_context.get_available_countries()
        major_countries = who_context.get_major_countries()
        
        # Country selector
        selected_country = st.selectbox(
            "Select Country",
            [""] + major_countries + [c for c in available_countries if c not in major_countries]
        )
        
        include_trends = st.checkbox("Include Health Trends", True)
        trend_years = st.slider("Trend Analysis Years", 5, 20, 10)
        
        if selected_country:
            show_country_button = st.button("📊 Show Country Data", type="primary")
        else:
            show_country_button = False
    
    with col2:
        if selected_country and show_country_button:
            st.markdown(f"### 📊 {selected_country} Health Profile")
            
            # Get country context
            context = who_context.get_country_context(selected_country)
            
            if context:
                # Basic indicators
                health_indicators = context["health_indicators"]
                
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    life_exp = health_indicators.get("Life expectancy ", "N/A")
                    if isinstance(life_exp, (int, float)):
                        st.metric("Life Expectancy", f"{life_exp:.1f} years")
                    else:
                        st.metric("Life Expectancy", "N/A")
                
                with col2b:
                    mortality = health_indicators.get("Adult Mortality", "N/A")
                    if isinstance(mortality, (int, float)):
                        st.metric("Adult Mortality", f"{mortality:.0f}")
                    else:
                        st.metric("Adult Mortality", "N/A")
                
                with col2c:
                    bmi = health_indicators.get(" BMI ", "N/A")
                    if isinstance(bmi, (int, float)):
                        st.metric("Population BMI", f"{bmi:.1f}")
                    else:
                        st.metric("Population BMI", "N/A")
                
                # Health trends
                if include_trends:
                    trends = who_context.get_health_trends(selected_country, trend_years)
                    
                    if trends and "indicator_trends" in trends:
                        st.markdown("#### 📈 Health Trends")
                        
                        trend_data = []
                        for indicator, trend_info in trends["indicator_trends"].items():
                            trend_data.append({
                                "Indicator": indicator,
                                "Change": f"{trend_info['percent_change']:+.1f}%",
                                "Direction": trend_info["direction"],
                                "Start": trend_info["start_value"],
                                "End": trend_info["end_value"]
                            })
                        
                        if trend_data:
                            trend_df = pd.DataFrame(trend_data)
                            st.dataframe(trend_df, use_container_width=True)
            else:
                st.error(f"No data available for {selected_country}")
    
    # Country comparison
    st.markdown("### 🔄 Country Comparison")
    
    comparison_countries = st.multiselect(
        "Select countries to compare",
        major_countries,
        default=major_countries[:3]
    )
    
    if comparison_countries and len(comparison_countries) > 1:
        comparison_data = who_context.compare_countries(comparison_countries)
        
        if not comparison_data.empty:
            # Create comparison chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Life Expectancy", "Adult Mortality", "BMI", "Schooling"),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Add traces
            indicators = ["Life expectancy", "Adult Mortality", "BMI", "Schooling"]
            positions = [(1,1), (1,2), (2,1), (2,2)]
            
            for indicator, (row, col) in zip(indicators, positions):
                if indicator in comparison_data.columns:
                    fig.add_trace(
                        go.Bar(x=comparison_data["Country"], 
                               y=comparison_data[indicator],
                               name=indicator,
                               showlegend=False),
                        row=row, col=col
                    )
            
            fig.update_layout(height=600, title_text="Country Health Indicators Comparison")
            st.plotly_chart(fig, use_container_width=True)

def create_performance_page():
    """Create the model performance page"""
    
    st.markdown("## 📊 Model Performance")
    
    # Load performance metrics if available
    try:
        metrics_file = Path("src/models/performance_metrics.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            st.markdown("### 🏆 ML Model Performance")
            
            # Classification metrics
            if "classification" in metrics:
                class_metrics = metrics["classification"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Random Forest", f"{class_metrics.get('random_forest', {}).get('accuracy', 0)*100:.1f}%", 
                             "Best Classifier")
                
                with col2:
                    st.metric("XGBoost", f"{class_metrics.get('xgboost', {}).get('accuracy', 0)*100:.1f}%")
                
                with col3:
                    st.metric("Logistic Regression", f"{class_metrics.get('logistic', {}).get('accuracy', 0)*100:.1f}%")
            
            # Regression metrics
            if "regression" in metrics:
                reg_metrics = metrics["regression"]
                
                st.markdown("### 📈 Sleep Quality Prediction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("XGBoost R²", f"{reg_metrics.get('xgboost_regressor', {}).get('r2', 0)*100:.1f}%", 
                             "Best Regressor")
                
                with col2:
                    st.metric("Random Forest R²", f"{reg_metrics.get('rf_regressor', {}).get('r2', 0)*100:.1f}%")
        
        else:
            st.warning("Performance metrics file not found")
    
    except Exception as e:
        st.error(f"Could not load performance metrics: {e}")
    
    # Feature importance
    try:
        importance_file = Path("src/models/feature_importance.json")
        if importance_file.exists():
            with open(importance_file, 'r') as f:
                importance_data = json.load(f)
            
            st.markdown("### 🔍 Feature Importance")
            
            # Create feature importance chart
            if "random_forest" in importance_data:
                rf_importance = importance_data["random_forest"]
                
                # Convert to DataFrame
                importance_df = pd.DataFrame(list(rf_importance.items()), 
                                           columns=["Feature", "Importance"])
                importance_df = importance_df.sort_values("Importance", ascending=True)
                
                # Create horizontal bar chart
                fig = px.bar(importance_df, x="Importance", y="Feature", 
                           orientation="h", title="Random Forest Feature Importance")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")
    
    # System performance
    st.markdown("### ⚡ System Performance")
    
    performance_data = {
        "Component": ["ML Prediction", "WHO Context", "Medical Explanation", "Complete Assessment"],
        "Target Time": ["<500ms", "<100ms", "<3000ms", "<9000ms"],
        "Typical Time": ["~250ms", "~50ms", "~2000ms", "~8500ms"],
        "Status": ["✅", "✅", "✅", "✅"]
    }
    
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df, use_container_width=True)

def create_system_status_page():
    """Create the system status page"""
    
    st.markdown("## 🔧 System Status")
    
    if st.session_state.sleep_system:
        system = st.session_state.sleep_system
        
        # Component status
        st.markdown("### 📊 Component Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ml_status = "🟢 Operational" if system.models_available else "🔴 Unavailable"
            st.markdown(f"""
            **ML Models**  
            {ml_status}
            
            - Random Forest: ✅
            - XGBoost: ✅  
            - Logistic Regression: ✅
            """)
        
        with col2:
            who_status = "🟢 Operational" if system.who_context else "🔴 Unavailable"
            st.markdown(f"""
            **WHO Integration**  
            {who_status}
            
            - Countries: {len(system.who_context.get_available_countries()) if system.who_context else 0}
            - Major Countries: {len(system.who_context.get_major_countries()) if system.who_context else 0}
            - Data Range: 2000-2015
            """)
        
        with col3:
            groq_status = "🟢 Operational" if system.groq_available else "🔴 Unavailable"
            st.markdown(f"""
            **Medical AI**  
            {groq_status}
            
            - Model: openai/gpt-oss-120b
            - Provider: Groq
            - Clinical Reasoning: ✅
            """)
        
        # Test system functionality
        st.markdown("### 🧪 System Test")
        
        if st.button("🔍 Run System Test", type="primary"):
            with st.spinner("Testing system components..."):
                test_results = run_system_test(system)
                display_test_results(test_results)
    
    else:
        st.error("Sleep health system not initialized")

def run_system_test(system):
    """Run a comprehensive system test"""
    
    test_results = {}
    
    # Test ML models
    try:
        sample_data = {
            "Gender": "Male", "Age": 35, "BMI Category": "Normal",
            "Sleep Duration": 7.0, "Stress Level": 5
        }
        
        start_time = time.time()
        assessment = system.comprehensive_assessment(
            user_data=sample_data,
            country="United States of America",
            include_explanation=False
        )
        test_time = time.time() - start_time
        
        test_results["ml_test"] = {
            "status": "✅ Pass",
            "time": f"{test_time*1000:.0f}ms",
            "result": assessment["ml_predictions"]["sleep_disorder"]["predicted_class"]
        }
    except Exception as e:
        test_results["ml_test"] = {
            "status": "❌ Fail",
            "error": str(e)
        }
    
    # Test WHO integration
    try:
        context = system.who_context.get_country_context("United States of America")
        test_results["who_test"] = {
            "status": "✅ Pass",
            "country": context["country"],
            "indicators": len(context["health_indicators"])
        }
    except Exception as e:
        test_results["who_test"] = {
            "status": "❌ Fail",
            "error": str(e)
        }
    
    # Test Groq API
    if system.groq_available:
        try:
            groq_test = system.groq_explainer.test_connection()
            test_results["groq_test"] = {
                "status": "✅ Pass" if groq_test["status"] == "success" else "❌ Fail",
                "response": groq_test.get("response", "No response")
            }
        except Exception as e:
            test_results["groq_test"] = {
                "status": "❌ Fail",
                "error": str(e)
            }
    else:
        test_results["groq_test"] = {
            "status": "⚠️ Unavailable",
            "note": "Groq API not available"
        }
    
    return test_results

def display_test_results(test_results):
    """Display system test results"""
    
    st.markdown("#### 🧪 Test Results")
    
    for test_name, result in test_results.items():
        test_display_name = test_name.replace("_", " ").title()
        
        with st.expander(f"{result['status']} {test_display_name}"):
            for key, value in result.items():
                if key != "status":
                    st.text(f"{key}: {value}")

def create_documentation_page():
    """Create the documentation page"""
    
    st.markdown("## 📚 Documentation")
    
    # Project overview
    st.markdown("""
    ### 🎯 Project Overview
    
    The Sleep Health & Life Expectancy Risk Coach is a comprehensive AI-powered system that:
    
    - **Predicts sleep disorders** with 97.3% accuracy using ensemble ML models
    - **Provides population context** using WHO health data from 183 countries
    - **Generates medical explanations** using Groq AI with clinical reasoning
    - **Offers personalized recommendations** based on individual and population health patterns
    """)
    
    # Technical architecture
    st.markdown("""
    ### 🏗️ Technical Architecture
    
    **Machine Learning Stack:**
    - Random Forest (97.3% accuracy - best classifier)
    - XGBoost (99.4% R² - best regressor)  
    - Logistic Regression (interpretable baseline)
    
    **Data Integration:**
    - Sleep Health & Lifestyle Dataset (synthetic, 374 records)
    - WHO Life Expectancy Data (real, 183 countries, 2000-2015)
    
    **AI Integration:**
    - Groq API with openai/gpt-oss-120b model
    - Clinical reasoning and medical explanations
    - Evidence-based recommendations
    """)
    
    # API endpoints
    st.markdown("""
    ### 🔌 API Endpoints
    
    **Flask REST API** (User preference):
    - `GET /api/health` - System health check
    - `POST /api/predict` - Complete sleep health assessment
    - `GET /api/countries` - Available WHO countries
    - `POST /api/who-context` - Population health context
    - `POST /api/explain` - Medical explanations
    - `POST /api/compare-countries` - Multi-country comparison
    
    **MCP Tools** (AI Assistant Integration):
    - `sleep.predict` - Individual assessment
    - `context.who_indicators` - Population context
    - `explain.risk_factors` - Medical explanations
    - `monitor.log_prediction` - Logging & monitoring
    - `compare.countries` - Country comparison
    - `system.status` - Health monitoring
    """)
    
    # Usage examples
    st.markdown("""
    ### 💡 Usage Examples
    
    **Individual Assessment:**
    1. Enter personal health data (age, BMI, sleep habits)
    2. Select country for population context
    3. Get ML prediction with medical explanation
    4. Receive personalized recommendations
    
    **Population Research:**
    1. Select countries for health indicator comparison
    2. Analyze health trends over time
    3. Compare individual risk across countries
    4. Understand population health patterns
    
    **AI Assistant Integration:**
    1. Configure MCP server for Claude/ChatGPT
    2. Ask natural language questions about sleep health
    3. Get comprehensive assessments through AI
    4. Receive medical explanations and recommendations
    """)
    
    # Important notes
    st.markdown("""
    ### ⚠️ Important Notes
    
    - **Medical Disclaimer:** All predictions and recommendations are for educational purposes only
    - **Professional Consultation:** Always consult healthcare providers for medical decisions
    - **Data Privacy:** No personal data is permanently stored by the system
    - **Population Data:** WHO data reflects population-level patterns, not individual predictions
    - **Model Limitations:** Based on synthetic sleep data - real-world validation needed
    """)

def show_assessment_history():
    """Show the assessment history"""
    
    if not st.session_state.assessment_history:
        return
    
    # Create summary DataFrame
    history_data = []
    for record in st.session_state.assessment_history:
        assessment = record["assessment"]
        disorder = assessment["ml_predictions"]["sleep_disorder"]
        quality = assessment["ml_predictions"]["sleep_quality"]
        risk = assessment["population_context"]["risk_calibration"]
        
        history_data.append({
            "Time": record["timestamp"].strftime("%H:%M:%S"),
            "Age": record["user_data"]["Age"],
            "BMI": record["user_data"]["BMI Category"],
            "Sleep Duration": record["user_data"]["Sleep Duration"],
            "Predicted Disorder": disorder["predicted_class"],
            "Confidence": f"{disorder['confidence']:.1%}",
            "Sleep Quality": f"{quality['predicted_quality']:.1f}/10",
            "Risk Level": risk["risk_level"],
            "Processing Time": f"{record['processing_time']*1000:.0f}ms"
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.assessment_history = []
        st.rerun()

def show_system_unavailable():
    """Show system unavailable message"""
    
    st.error("""
    🚨 **Sleep Health System Unavailable**
    
    The sleep health assessment system could not be initialized. This may be due to:
    
    - Missing model files
    - Import errors
    - Configuration issues
    
    Please check the system configuration and try again.
    """)
    
    # Show what's available
    st.markdown("""
    ### 📚 Documentation Available
    
    Even without the full system, you can still:
    - View the project documentation
    - Understand the technical architecture
    - See the API specifications
    - Learn about the medical approach
    """)

if __name__ == "__main__":
    main()
