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
   cd ~/New-Life
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


**Status**: 🎉 **PROJECT COMPLETE** ✅ | Production Ready 🚀 | **Ready for Deployment** 🌟
