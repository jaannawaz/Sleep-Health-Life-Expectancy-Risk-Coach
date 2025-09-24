# Sleep Health & Life Expectancy Risk Coach (MCP-Integrated)

A real-time AI health application that predicts sleep disorder risk and sleep quality using synthetic sleep data and contextualizes predictions with WHO Life Expectancy data. The system exposes functionality through the Model Context Protocol (MCP) for seamless integration with AI assistants.

## 🎯 Project Overview

This application combines:
- **Individual-level predictions** using Sleep Health & Lifestyle dataset (synthetic)
- **Population-level context** using WHO Life Expectancy data (real)
- **Medical explanations** via Groq API integration (fast LLM inference)
- **MCP tools** for AI assistant integration

## 📊 Datasets

### Sleep Health & Lifestyle Dataset (Synthetic)
- **Size**: 374 records, 13 features
- **Target Variables**: 
  - Sleep Disorder (binary classification): Sleep Apnea (78), Insomnia (77), None (219)
  - Quality of Sleep (regression): 1-10 scale
- **Key Features**: Age, Gender, Occupation, Sleep Duration, Physical Activity, Stress Level, BMI Category, Heart Rate, Daily Steps

### WHO Life Expectancy Data (Real)
- **Size**: 2,938 records, 22 features
- **Coverage**: 193 countries, 2000-2015
- **Key Indicators**: Life Expectancy, Adult Mortality, BMI, GDP, Income Composition, Schooling
- **Development Status**: Developing (1,680) vs Developed (1,258) countries

## 🏗️ Project Structure

```
New-Life/
├── config/
│   └── config.py                 # Project configuration
├── data/
│   ├── raw/                      # Original CSV datasets
│   └── processed/                # Cleaned and harmonized data
├── src/
│   ├── data/                     # Data processing scripts
│   │   └── who_integration.py    # WHO health context & population benchmarking
│   ├── models/                   # ML model implementations
│   │   ├── ensemble_models.py    # XGBoost + RF + Logistic ensemble
│   │   ├── test_ensemble.py      # Model testing and validation
│   │   ├── visualize_results.py  # Comprehensive visualization generator
│   │   ├── sleep_who_integration.py # Complete ML + WHO integration system
│   │   ├── quick_integration_test.py # Integration validation tests
│   │   └── complete_sleep_health_system.py # Full ML + WHO + Groq integration
│   ├── api/                      # API endpoints and integrations
│   │   └── groq_explainer.py     # Groq medical explanation engine
│   ├── mcp_tools/               # MCP tool implementations (Production Ready)
│   │   ├── sleep_health_mcp_server.py # Main MCP server with 6 tools
│   │   ├── tool_schemas.py       # Input/output schemas and validation
│   │   ├── test_mcp_client.py    # Comprehensive test suite (22/22 tests pass)
│   │   ├── mcp_demo.py          # Integration demonstration
│   │   ├── mcp_documentation.json # Complete API documentation
│   │   └── README.md            # MCP tools usage guide
│   └── utils/                    # Utility functions
├── notebooks/
│   └── 01_data_exploration.ipynb # Data exploration notebook
├── visualizations/              # Model performance charts & graphs
│   ├── 00_executive_summary.png  # Complete performance overview
│   ├── 01_model_performance_comparison.png
│   ├── 02_feature_importance.png
│   ├── 03_confusion_matrices.png
│   ├── 04_roc_curves.png
│   ├── 05_prediction_distributions.png
│   ├── 06_cross_validation_curves.png
│   └── README.md                 # Visualization documentation
├── logs/                         # Application logs
├── venv/                         # Virtual environment
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation Steps

1. **Clone and navigate to project**:
   ```bash
   cd /Users/jaan/Desktop/New-Life
   ```

2. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Dependencies are already installed**, but if you need to reinstall:
   ```bash
   pip install -r requirements.txt
   ```

4. **Test the setup**:
   ```bash
   python -c "
   import sys; sys.path.append('config')
   from config import *
   import pandas as pd
   sleep_df = pd.read_csv(SLEEP_DATA_FILE)
   life_df = pd.read_csv(LIFE_EXPECTANCY_DATA_FILE)
   print(f'✓ Setup complete! Sleep: {sleep_df.shape}, Life: {life_df.shape}')
   "
   ```

## 📋 Development Roadmap

### ✅ Completed (Week 1)
- [x] Virtual environment setup with all dependencies
- [x] Project structure and configuration
- [x] **Data exploration and quality assessment**
- [x] **Dataset integration strategy**
- [x] **Data cleaning & harmonization**
- [x] **Risk factor analysis and insights**

### ✅ Current Progress (Week 2) - COMPLETED
- [x] **ML Strategy**: Ensemble approach with XGBoost focus
- [x] **Feature Engineering**: Categorical encoding and scaling
- [x] **Model Development**: XGBoost + Random Forest + Logistic Regression
- [x] **Model Evaluation**: Cross-validation and performance comparison
- [x] **Sleep Quality Regression**: Secondary prediction target

### 🏆 Ensemble Performance Results
- **Random Forest**: 97.3% accuracy (best classifier)
- **XGBoost**: 94.7% accuracy, 99.4% R² (best regressor)
- **Logistic Regression**: 93.3% accuracy (interpretable baseline)
- **Key Features**: BMI Category (57.6%), Blood Pressure (13.9%)
- **All models**: Excellent performance with minimal overfitting

### 📊 Comprehensive Visualizations
**7 detailed charts and graphs** available in `visualizations/` directory:
- **Executive Summary**: Complete performance overview and medical insights
- **Performance Comparison**: Side-by-side model accuracy, F1, and R² scores
- **Feature Importance**: BMI Category (57.6%) and Blood Pressure (13.9%) as top predictors
- **Confusion Matrices**: Error analysis for all 3 sleep disorder classes
- **ROC Curves**: 99%+ AUC scores demonstrating excellent discrimination
- **Prediction Distributions**: Risk patterns by BMI, age, and sleep duration
- **Cross-Validation Curves**: Learning behavior and overfitting analysis

### ✅ Week 3 Progress - COMPLETED  
- [x] **WHO Data Integration**: Population health context system
- [x] **Country Benchmarking**: Individual vs population risk calibration
- [x] **BMI Harmonization**: Sleep dataset ↔ WHO dataset mapping
- [x] **Health Trends Analysis**: Temporal country health patterns
- [x] **Risk Calibration Framework**: Population-adjusted predictions

### ✅ Week 4 Progress - COMPLETED
- [x] **Groq API Integration**: Fast LLM inference for medical explanations  
- [x] **Clinical Reasoning Engine**: Evidence-based sleep health analysis
- [x] **Medical Explanation System**: Clinical-style risk factor interpretation
- [x] **Enhanced Recommendations**: AI-powered personalized health guidance
- [x] **Complete System Integration**: ML + WHO + Groq unified platform

### ✅ Week 5 Progress - COMPLETED
- [x] **MCP Server Implementation**: Full Model Context Protocol integration
- [x] **6 MCP Tools**: sleep.predict, context.who_indicators, explain.risk_factors, monitor.log_prediction, compare.countries, system.status
- [x] **Input Validation**: Comprehensive schema validation and error handling
- [x] **AI Assistant Ready**: Compatible with Claude, ChatGPT, and other MCP systems
- [x] **Production Monitoring**: Logging, performance tracking, and health monitoring

### ✅ Week 6 Progress - COMPLETED
- [x] **Flask REST API**: 8 comprehensive endpoints with full documentation
- [x] **Streamlit Dashboard**: Interactive web application with real-time assessment
- [x] **Docker Deployment**: Multi-container production configuration
- [x] **Deployment Automation**: Scripts for local development and production  
- [x] **Cloud-Ready Configuration**: AWS/GCP/Azure deployment support
- [x] **Final Documentation**: Complete deployment guides and project summary

### 🎉 Project Status: **100% COMPLETE**
All 6 weekly milestones achieved successfully! Ready for production deployment.

## 🛠️ Core Features

### ✅ MCP Tools (Production Ready)
1. **`sleep.predict`**: Individual sleep disorder risk and quality prediction (97.3% accuracy)
2. **`context.who_indicators`**: Country-level health context from WHO data (183 countries)
3. **`explain.risk_factors`**: Medical-style explanations via Groq API (clinical reasoning)
4. **`monitor.log_prediction`**: Prediction logging for drift detection (real-time monitoring)
5. **`compare.countries`**: Multi-country risk comparison (up to 10 countries)
6. **`system.status`**: System health and component availability monitoring

### API Endpoints
- REST API for direct health predictions
- Streamlit dashboard for interactive exploration
- MCP server for AI assistant integration

### Performance Targets
- Response time: <500ms (including Groq API call)
- Model accuracy: >75% for sleep disorder classification
- Reliability: Graceful fallbacks when external APIs unavailable

## 🤖 Machine Learning Approach

### Ensemble Strategy
We implement a **multi-model ensemble approach** to balance performance and interpretability:

#### Primary Models:
1. **🚀 XGBoost Classifier**
   - High-performance gradient boosting
   - Excellent for tabular data with mixed features
   - Built-in regularization prevents overfitting
   - Feature importance for medical insights

2. **🌲 Random Forest Classifier**
   - Robust baseline with interpretability
   - Feature importance rankings
   - Less prone to overfitting on small datasets
   - Good benchmark for comparison

3. **📈 Logistic Regression**
   - Medical standard for probability interpretation
   - Linear coefficients for feature understanding
   - Fast inference for real-time predictions
   - Baseline for statistical significance

#### Model Selection Strategy:
- **Cross-validation** (k=5) for reliable evaluation on 374 records
- **Performance metrics**: Accuracy, F1-score, ROC-AUC, Precision, Recall
- **Feature importance** analysis across all models
- **Error analysis** for medical insight validation
- **Final selection**: Best performer or voting ensemble

#### Target Variables:
1. **Sleep Disorder Classification** (3 classes):
   - None: 219 records (58.6%)
   - Sleep Apnea: 78 records (20.9%) 
   - Insomnia: 77 records (20.6%)

2. **Sleep Quality Regression** (1-10 scale):
   - Mean: 7.3, Range: 4-9
   - Secondary target for quality prediction

## 📊 Data Integration Strategy

### Individual Risk Prediction (Sleep Dataset)
```python
# Input features
user_data = {
    "age": 28,
    "gender": "Male", 
    "sleep_duration": 6.1,
    "physical_activity": 42,
    "stress_level": 6,
    "bmi_category": "Overweight"
}

# Predictions
sleep_disorder_risk = model.predict_disorder(user_data)
sleep_quality_score = model.predict_quality(user_data)
```

### Population Context (WHO Dataset)
```python
# Country health indicators
country_context = who_data.get_indicators(
    country="United States of America",
    year=2015
)
# Returns: life_expectancy, adult_mortality, avg_bmi, etc.
```

### Medical Explanations (Groq API Integration)
```python
# Clinical-style reasoning with fast LLM inference
explanation = groq_explainer.explain_risk(
    prediction=sleep_disorder_risk,
    user_factors=user_data,
    population_context=country_context,
    model="openai/gpt-oss-120b"
)
```

## 🔬 Key Insights from Data Exploration

### Sleep Health Dataset
- **Clean data**: No missing values, ready for modeling
- **Balanced targets**: 58% no disorder, 21% Sleep Apnea, 21% Insomnia
- **Key risk factors**: Lower sleep duration, higher stress, specific BMI categories
- **Age range**: 27-59 years (working adults)

### WHO Life Expectancy Dataset
- **Global coverage**: 193 countries over 16 years (2000-2015)
- **Missing data**: ~10% overall, varies by indicator and country
- **Trends**: Generally improving life expectancy and health indicators
- **Development gap**: Clear differences between developed/developing countries

## 🤝 Integration Opportunities

1. **BMI harmonization**: Map categorical BMI (sleep) to numeric BMI (WHO)
2. **Country context**: Allow users to select country for population comparison
3. **Age-adjusted predictions**: Use WHO adult mortality rates for age contextualization
4. **Lifestyle benchmarking**: Compare individual risk to population health trends

## 📝 Next Steps

1. **Current Phase - ML Model Development**: 
   - Install XGBoost dependency
   - Implement feature engineering pipeline
   - Build ensemble models (XGBoost + Random Forest + Logistic Regression)
   - Cross-validation and performance evaluation
   - Feature importance analysis for medical insights

2. **Upcoming Phase - Integration & APIs**:
   - WHO dataset integration for country context
   - FastAPI backend development 
   - MCP tools implementation
   - Streamlit dashboard for interactive predictions

## 🔍 Key Risk Insights Discovered

Our data analysis revealed critical patterns for model development:

### Sleep Disorder Risk Factors:
- **BMI Impact**: Obese individuals show 100% sleep disorder risk
- **Weight Categories**: Overweight (87.2% risk) vs Normal (7.4% risk)
- **Age Correlation**: Risk increases from 15% (young) to 65% (older adults)
- **Sleep Duration**: Short sleepers have 68% disorder risk vs 30% normal duration
- **Quality Correlation**: Lower sleep quality strongly predicts disorders

### WHO Health Trends (2000-2015):
- **Life Expectancy**: +7.3% global improvement
- **Adult Mortality**: -15.8% reduction (positive trend)
- **BMI Concern**: +24.2% increase (potential obesity epidemic)
- **Education**: +23% improvement in schooling years

## 📚 Technical Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: XGBoost, scikit-learn (Random Forest, Logistic Regression)
- **API Framework**: Flask (per user preference)
- **Dashboard**: Streamlit
- **MCP Integration**: mcp library
- **External APIs**: Groq (fast LLM inference)
- **Visualization**: matplotlib, seaborn
- **Model Evaluation**: cross-validation, sklearn.metrics
- **Feature Engineering**: sklearn.preprocessing, pandas.get_dummies

## ⚠️ Important Notes

- This is an **educational/demo application only**
- Not intended for actual medical diagnosis or advice
- Synthetic sleep data may not reflect real patient populations
- WHO data is population-level, not individual-level
- All predictions should include appropriate disclaimers

---

**Status**: 🎉 **PROJECT COMPLETE** ✅ | All 6 Milestones Achieved | Production Ready 🚀 | **Ready for Deployment** 🌟
