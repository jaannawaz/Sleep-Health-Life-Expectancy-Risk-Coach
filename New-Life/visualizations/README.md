# Sleep Health Ensemble Model Visualizations

This directory contains comprehensive visualizations for the Sleep Health & Life Expectancy Risk Coach ensemble models.

## 📊 Generated Charts & Graphs

### 1. **Executive Summary Dashboard** (`00_executive_summary.png`)
- **Overview**: Complete performance summary and key insights
- **Contents**: Model accuracies, R² scores, feature importance highlights, medical insights
- **Use Case**: Executive presentation, project overview

### 2. **Model Performance Comparison** (`01_model_performance_comparison.png`)
- **Overview**: Side-by-side comparison of all ensemble models
- **Contents**: 
  - Classification accuracy (XGBoost: 94.7%, Random Forest: 97.3%, Logistic: 93.3%)
  - F1 scores for sleep disorder prediction
  - RMSE scores for sleep quality regression (XGBoost: 0.093, RF: 0.138)
  - R² scores (XGBoost: 99.4%, RF: 98.7%)
- **Use Case**: Model selection, performance benchmarking

### 3. **Feature Importance Analysis** (`02_feature_importance.png`)
- **Overview**: Critical features driving sleep disorder predictions
- **Contents**:
  - XGBoost importance: BMI Category (57.6%), Blood Pressure (13.9%)
  - Random Forest importance rankings
  - Top 10 most influential features
- **Use Case**: Medical interpretation, feature selection

### 4. **Confusion Matrices** (`03_confusion_matrices.png`)
- **Overview**: Classification accuracy breakdown by sleep disorder type
- **Contents**:
  - 3x3 confusion matrices for each model (None, Insomnia, Sleep Apnea)
  - Normalized percentages and raw counts
  - True vs predicted label analysis
- **Use Case**: Error analysis, class-specific performance

### 5. **ROC Curves** (`04_roc_curves.png`)
- **Overview**: Receiver Operating Characteristic curves for multi-class classification
- **Contents**:
  - ROC curves for each sleep disorder class
  - Area Under Curve (AUC) scores (>99% for all models)
  - False positive vs true positive rates
- **Use Case**: Model discrimination ability, threshold selection

### 6. **Prediction Distributions** (`05_prediction_distributions.png`)
- **Overview**: Data distribution analysis and relationship exploration
- **Contents**:
  - Sleep disorder distribution pie chart (58.6% None, 20.9% Sleep Apnea, 20.6% Insomnia)
  - Sleep quality score histogram
  - BMI category vs sleep disorder risk analysis
  - Sleep duration vs quality scatter plot with trend line
- **Use Case**: Data understanding, risk factor visualization

### 7. **Cross-Validation Learning Curves** (`06_cross_validation_curves.png`)
- **Overview**: Model learning behavior and overfitting analysis
- **Contents**:
  - Training vs validation score curves for all models
  - Performance stability across different training set sizes
  - Confidence intervals showing model reliability
- **Use Case**: Overfitting detection, training optimization

## 🎯 Key Performance Highlights

| Model | Accuracy | F1-Score | ROC-AUC | RMSE | R² |
|-------|----------|----------|---------|------|----|
| **Random Forest** | **97.3%** | **97.3%** | **99.8%** | 0.138 | 98.7% |
| **XGBoost** | 94.7% | 94.7% | 99.4% | **0.093** | **99.4%** |
| **Logistic Regression** | 93.3% | 93.3% | 99.0% | - | - |

## 🔍 Medical Insights from Visualizations

### High-Risk Factors (from Feature Importance):
1. **BMI Category** (57.6% importance): Strongest predictor
   - Obese: 100% sleep disorder risk
   - Overweight: 87.2% risk
   - Normal: 7.4% risk

2. **Blood Pressure** (13.9% importance): Cardiovascular correlation
3. **Occupation** (7.4% importance): Lifestyle impact
4. **Heart Rate** (4.7% importance): Physiological indicator

### Age-Related Risk Patterns:
- Young adults (27-35): 15% risk
- Middle-aged (35-45): 41% risk  
- Older adults (45+): 65% risk

### Sleep Duration Impact:
- Short sleepers (<6.5h): 68% disorder risk
- Normal duration (6.5-7.5h): 30% risk
- Long sleepers (>7.5h): 27% risk

## 📈 Technical Quality Metrics

- **Cross-Validation**: Consistent performance across all folds
- **Learning Curves**: No significant overfitting detected
- **ROC Analysis**: Excellent discrimination (AUC > 99%)
- **Confusion Matrices**: High precision and recall across all classes
- **Feature Stability**: Consistent importance rankings across models

## 🚀 Usage for Development

These visualizations support:
- **Model Selection**: Random Forest for classification, XGBoost for regression
- **Feature Engineering**: Focus on BMI, blood pressure, and lifestyle factors
- **API Development**: Confidence thresholds and prediction interpretation
- **UI/Dashboard**: Risk visualization and user feedback
- **Medical Validation**: Clinical relevance and interpretability

## 📊 File Specifications

- **Format**: High-resolution PNG (300 DPI)
- **Size**: ~2.5MB total, ~350KB average per chart
- **Dimensions**: Optimized for both screen and print display
- **Quality**: Production-ready for presentations and documentation

---

**Generated**: September 24, 2025  
**Models**: XGBoost 3.0.5, scikit-learn 1.7.2, ensemble approach  
**Dataset**: 374 samples, 12 features, 3-class + regression targets
