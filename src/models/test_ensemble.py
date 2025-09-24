"""
Test script to demonstrate the Sleep Health Ensemble Models
"""

import pandas as pd
import joblib
import json
from pathlib import Path
import sys
import os

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))
try:
    from config import *
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "src" / "models"

from ensemble_models import SleepHealthEnsemble

def load_trained_ensemble():
    """Load the trained ensemble models"""
    
    ensemble = SleepHealthEnsemble(random_state=42)
    
    # Load models
    model_files = {
        'xgboost': 'xgboost_model.joblib',
        'random_forest': 'random_forest_model.joblib', 
        'logistic': 'logistic_model.joblib',
        'xgboost_regressor': 'xgboost_regressor_model.joblib',
        'rf_regressor': 'rf_regressor_model.joblib'
    }
    
    for model_name, filename in model_files.items():
        model_path = MODELS_DIR / filename
        if model_path.exists():
            ensemble.models[model_name] = joblib.load(model_path)
            print(f"✓ Loaded {model_name}")
    
    # Load preprocessors
    preprocessor_path = MODELS_DIR / "preprocessors.joblib"
    if preprocessor_path.exists():
        ensemble.preprocessors = joblib.load(preprocessor_path)
        ensemble.regression_preprocessors = ensemble.preprocessors.copy()  # For simplicity
        
        # Set feature names (these would normally be set during training)
        sample_features = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
                          'Physical Activity Level', 'Stress Level', 'BMI Category', 
                          'Blood Pressure', 'Heart Rate', 'Daily Steps']
        ensemble.feature_names = sample_features
        ensemble.regression_feature_names = [f for f in sample_features if f != 'Quality of Sleep']
        
        print(f"✓ Loaded preprocessors")
    
    # Load performance metrics
    metrics_path = MODELS_DIR / "performance_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            ensemble.performance_metrics = json.load(f)
        print(f"✓ Loaded performance metrics")
    
    return ensemble

def test_predictions():
    """Test the ensemble with sample user data"""
    
    print("🚀 Testing Sleep Health Ensemble Models")
    print("=" * 60)
    
    # Load ensemble
    ensemble = load_trained_ensemble()
    
    # Test cases representing different risk profiles
    test_cases = [
        {
            "name": "High Risk User",
            "data": {
                "Gender": "Male",
                "Age": 45,
                "Occupation": "Software Engineer", 
                "Sleep Duration": 5.5,
                "Physical Activity Level": 30,
                "Stress Level": 8,
                "BMI Category": "Obese",
                "Blood Pressure": "140/90",
                "Heart Rate": 85,
                "Daily Steps": 3000,
                "Quality of Sleep": 4  # Low quality for high risk
            }
        },
        {
            "name": "Low Risk User", 
            "data": {
                "Gender": "Female",
                "Age": 30,
                "Occupation": "Teacher",
                "Sleep Duration": 7.5,
                "Physical Activity Level": 60,
                "Stress Level": 4,
                "BMI Category": "Normal",
                "Blood Pressure": "120/80",
                "Heart Rate": 70,
                "Daily Steps": 8000,
                "Quality of Sleep": 8  # High quality for low risk
            }
        },
        {
            "name": "Medium Risk User",
            "data": {
                "Gender": "Male",
                "Age": 38,
                "Occupation": "Lawyer",
                "Sleep Duration": 6.2,
                "Physical Activity Level": 45,
                "Stress Level": 6,
                "BMI Category": "Overweight", 
                "Blood Pressure": "130/85",
                "Heart Rate": 75,
                "Daily Steps": 5500,
                "Quality of Sleep": 6  # Medium quality for medium risk
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\\n🧪 Testing: {test_case['name']}")
        print("-" * 40)
        
        user_data = test_case['data']
        
        # Display user profile
        print("User Profile:")
        for key, value in user_data.items():
            print(f"  {key:25} -> {value}")
        
        print("\\nPredictions:")
        
        # Test sleep disorder prediction with all models
        for model_name in ['xgboost', 'random_forest', 'logistic']:
            try:
                result = ensemble.predict_sleep_disorder(user_data, model_name)
                predicted_class = result['predicted_class']
                confidence = result['confidence']
                
                print(f"  {model_name.replace('_', ' ').title():15} -> {predicted_class:12} (confidence: {confidence:.1%})")
                
                if model_name == 'xgboost':  # Show probabilities for XGBoost
                    probs = result['probabilities']
                    for disorder, prob in probs.items():
                        print(f"    {disorder:12} -> {prob:.1%}")
                        
            except Exception as e:
                print(f"  {model_name:15} -> Error: {e}")
        
        # Test sleep quality prediction
        try:
            quality_result = ensemble.predict_sleep_quality(user_data, 'xgboost_regressor')
            quality_score = quality_result['predicted_quality']
            quality_level = quality_result['quality_level']
            print(f"\\n  Sleep Quality Score: {quality_score:.1f}/10 ({quality_level})")
        except Exception as e:
            print(f"\\n  Sleep Quality -> Error: {e}")
        
        print()

def display_model_performance():
    """Display model performance metrics"""
    
    print("📊 Model Performance Summary")
    print("=" * 60)
    
    metrics_path = MODELS_DIR / "performance_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        if 'classification' in metrics:
            print("\\nClassification Models (Sleep Disorder Prediction):")
            print("-" * 50)
            
            for model_name, results in metrics['classification'].items():
                print(f"{model_name.replace('_', ' ').title():15}:")
                print(f"  Accuracy: {results['test_accuracy']:.1%}")
                print(f"  F1-Score: {results['test_f1']:.1%}")
                if results['test_auc']:
                    print(f"  ROC-AUC:  {results['test_auc']:.1%}")
                print(f"  CV Score: {results['cv_mean']:.1%} (±{results['cv_std']:.1%})")
                print()
        
        if 'regression' in metrics:
            print("Regression Models (Sleep Quality Prediction):")
            print("-" * 50)
            
            for model_name, results in metrics['regression'].items():
                display_name = model_name.replace('_regressor', '').replace('_', ' ').title()
                print(f"{display_name:15}:")
                print(f"  RMSE:     {results['test_rmse']:.3f}")
                print(f"  MAE:      {results['test_mae']:.3f}")
                print(f"  R²:       {results['test_r2']:.1%}")
                print(f"  CV RMSE:  {results['cv_rmse_mean']:.3f} (±{results['cv_rmse_std']:.3f})")
                print()
    
    # Feature importance
    importance_path = MODELS_DIR / "feature_importance.json"
    if importance_path.exists():
        with open(importance_path) as f:
            importance = json.load(f)
        
        print("Top Feature Importance (XGBoost):")
        print("-" * 50)
        
        if 'xgboost' in importance:
            sorted_features = sorted(importance['xgboost'].items(), 
                                   key=lambda x: x[1], reverse=True)
            
            for i, (feature, imp) in enumerate(sorted_features[:8]):
                print(f"  {i+1:2d}. {feature:25} -> {imp:.1%}")

def main():
    """Main test function"""
    
    # Check if models exist
    if not (MODELS_DIR / "xgboost_model.joblib").exists():
        print("❌ Models not found! Please run ensemble_models.py first.")
        return
    
    # Display performance
    display_model_performance()
    
    # Test predictions
    test_predictions()
    
    print("🎉 Ensemble testing complete!")
    print(f"Models available at: {MODELS_DIR}")
    print("Ready for MCP integration and API development!")

if __name__ == "__main__":
    main()
