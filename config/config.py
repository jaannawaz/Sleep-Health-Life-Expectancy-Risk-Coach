"""
Configuration file for Sleep Health & Life Expectancy Risk Coach
"""
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data files
SLEEP_DATA_FILE = RAW_DATA_DIR / "Sleep_health_and_lifestyle_dataset.csv"
LIFE_EXPECTANCY_DATA_FILE = RAW_DATA_DIR / "Life Expectancy Data.csv"

# Model configuration
MODEL_CONFIG = {
    "sleep_disorder_classifier": {
        "target_column": "Sleep Disorder",
        "feature_columns": ["Age", "Sleep Duration", "Quality of Sleep", 
                           "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps"],
        "categorical_columns": ["Gender", "Occupation", "BMI Category"],
        "test_size": 0.2,
        "random_state": 42
    },
    "sleep_quality_regressor": {
        "target_column": "Quality of Sleep",
        "feature_columns": ["Age", "Sleep Duration", "Physical Activity Level", 
                           "Stress Level", "Heart Rate", "Daily Steps"],
        "categorical_columns": ["Gender", "Occupation", "BMI Category"],
        "test_size": 0.2,
        "random_state": 42
    }
}

# MCP Tools configuration
MCP_CONFIG = {
    "server_name": "sleep-health-coach",
    "tools": {
        "sleep_predict": {
            "name": "predict_sleep_risk",
            "description": "Predict sleep disorder risk and sleep quality based on lifestyle factors"
        },
        "who_context": {
            "name": "get_country_health_context", 
            "description": "Get WHO health indicators for a specific country"
        },
        "explain_risk": {
            "name": "explain_risk_factors",
            "description": "Explain sleep risk factors using medical reasoning"
        },
        "monitor_prediction": {
            "name": "log_prediction",
            "description": "Log prediction for monitoring and drift detection"
        }
    }
}

# API Configuration
API_CONFIG = {
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY", ""),  # Read from environment; keep empty by default
        "model": "openai/gpt-oss-120b",
        "temperature": 0.3,  # Lower for medical consistency
        "max_completion_tokens": 2048,
        "timeout": 30,
        "max_retries": 3
    },
    "flask": {
        "host": "localhost",
        "port": 5000,
        "debug": True
    },
    "streamlit": {
        "port": 8501
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "app.log"
}

# WHO dataset configuration
WHO_CONFIG = {
    "key_indicators": [
        "Life expectancy ", 
        "Adult Mortality", 
        " BMI ", 
        "Income composition of resources", 
        "Schooling",
        "GDP",
        "Population"
    ],
    "default_year": 2015,  # Latest year with good coverage
    "major_countries": [
        "United States of America", "United Kingdom", "Germany", "Japan",
        "Australia", "Canada", "France", "Italy", "Spain", "Netherlands",
        "Sweden", "Norway", "Denmark", "Finland", "Switzerland"
    ]
}

# Performance thresholds (as per PRD)
PERFORMANCE_CONFIG = {
    "response_time_ms": 500,
    "model_accuracy_threshold": 0.75,
    "prediction_confidence_threshold": 0.6
}
