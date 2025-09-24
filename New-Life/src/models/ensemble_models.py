"""
Ensemble ML Models for Sleep Health Prediction
Implements XGBoost + Random Forest + Logistic Regression approach
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))
try:
    from config import *
except ImportError:
    # Fallback if config import fails
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "src" / "models"
    MODELS_DIR.mkdir(exist_ok=True)

class SleepHealthEnsemble:
    """
    Ensemble model for sleep health prediction combining:
    - XGBoost (primary high-performance model)
    - Random Forest (interpretable baseline)
    - Logistic Regression (medical standard)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.preprocessors = {}
        self.regression_preprocessors = {}
        self.performance_metrics = {}
        self.feature_names = None
        self.regression_feature_names = None
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all models with optimal hyperparameters"""
        
        # XGBoost Classifier (primary model)
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        
        # Random Forest Classifier (interpretable baseline)
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )
        
        # Logistic Regression (medical standard)
        self.models['logistic'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            multi_class='multinomial'
        )
        
        # XGBoost Regressor for sleep quality
        self.models['xgboost_regressor'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state
        )
        
        # Random Forest Regressor for sleep quality
        self.models['rf_regressor'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )
    
    def preprocess_data(self, df, target_col='Sleep Disorder', fit_preprocessors=True, use_regression_preprocessors=False):
        """
        Preprocess the dataset for ML training
        
        Args:
            df: DataFrame with raw data
            target_col: Target column name
            fit_preprocessors: Whether to fit preprocessors (True for training)
            use_regression_preprocessors: Use separate preprocessors for regression
            
        Returns:
            X_processed, y: Processed features and target
        """
        
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # Separate features and target
        if target_col in data.columns:
            y = data[target_col].copy()
            X = data.drop([target_col, 'Person ID'], axis=1)
        else:
            y = None
            X = data.drop(['Person ID'], axis=1, errors='ignore')
        
        # Handle missing values in target (convert NaN to 'None')
        if y is not None and y.dtype == 'object':
            y = y.fillna('None')
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Handle categorical variables
        X_processed = X.copy()
        
        # Choose which preprocessors to use
        if use_regression_preprocessors:
            preprocessors = self.regression_preprocessors
        else:
            preprocessors = self.preprocessors
        
        if fit_preprocessors:
            preprocessors['label_encoders'] = {}
            preprocessors['scaler'] = StandardScaler()
        
        # Encode categorical variables
        for col in categorical_cols:
            if fit_preprocessors:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                preprocessors['label_encoders'][col] = le
            else:
                if col in preprocessors['label_encoders']:
                    le = preprocessors['label_encoders'][col]
                    # Handle unseen categories
                    X_processed[col] = X_processed[col].astype(str)
                    mask = X_processed[col].isin(le.classes_)
                    X_processed.loc[~mask, col] = le.classes_[0]  # Default to first class
                    X_processed[col] = le.transform(X_processed[col])
        
        # Scale numerical features
        if len(numerical_cols) > 0:
            if fit_preprocessors:
                X_processed[numerical_cols] = preprocessors['scaler'].fit_transform(X_processed[numerical_cols])
            else:
                X_processed[numerical_cols] = preprocessors['scaler'].transform(X_processed[numerical_cols])
        
        # Encode target variable for classification
        if y is not None and target_col == 'Sleep Disorder':
            if fit_preprocessors:
                self.preprocessors['target_encoder'] = LabelEncoder()
                y_encoded = self.preprocessors['target_encoder'].fit_transform(y)
            else:
                y_encoded = self.preprocessors['target_encoder'].transform(y)
            
            self.feature_names = X_processed.columns.tolist()
            return X_processed, y_encoded
        
        # Set feature names based on type
        if use_regression_preprocessors:
            self.regression_feature_names = X_processed.columns.tolist()
        else:
            self.feature_names = X_processed.columns.tolist()
            
        return X_processed, y
    
    def train_classification_models(self, df):
        """Train all classification models for sleep disorder prediction"""
        
        print("🔄 Training Classification Models for Sleep Disorder Prediction")
        print("=" * 70)
        
        # Preprocess data
        X, y = self.preprocess_data(df, target_col='Sleep Disorder', fit_preprocessors=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train each model and evaluate
        classification_results = {}
        
        for model_name in ['xgboost', 'random_forest', 'logistic']:
            print(f"\\n🤖 Training {model_name.replace('_', ' ').title()}...")
            
            model = self.models[model_name]
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='accuracy'
            )
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Multi-class ROC AUC
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                auc = None
            
            classification_results[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_f1': f1,
                'test_auc': auc,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, 
                                                              target_names=self.preprocessors['target_encoder'].classes_,
                                                              output_dict=True)
            }
            
            print(f"   CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"   Test Accuracy: {accuracy:.3f}")
            print(f"   Test F1: {f1:.3f}")
            if auc:
                print(f"   Test AUC: {auc:.3f}")
        
        self.performance_metrics['classification'] = classification_results
        
        # Find best model
        best_model = max(classification_results.keys(), 
                        key=lambda x: classification_results[x]['test_accuracy'])
        
        print(f"\\n🏆 Best Classification Model: {best_model.replace('_', ' ').title()}")
        print(f"   Accuracy: {classification_results[best_model]['test_accuracy']:.3f}")
        
        return classification_results
    
    def train_regression_models(self, df):
        """Train regression models for sleep quality prediction"""
        
        print("\\n🔄 Training Regression Models for Sleep Quality Prediction")
        print("=" * 70)
        
        # Create a separate copy for regression preprocessing
        df_regression = df.copy()
        
        # Preprocess data for regression (fit new preprocessors without Quality of Sleep)
        X, y = self.preprocess_data(df_regression, target_col='Quality of Sleep', fit_preprocessors=True, use_regression_preprocessors=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        regression_results = {}
        
        for model_name in ['xgboost_regressor', 'rf_regressor']:
            display_name = model_name.replace('_regressor', '').replace('_', ' ').title()
            print(f"\\n🤖 Training {display_name} Regressor...")
            
            model = self.models[model_name]
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            regression_results[model_name] = {
                'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
                'cv_rmse_std': np.sqrt(cv_scores.std()),
                'test_mse': mse,
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2
            }
            
            print(f"   CV RMSE: {np.sqrt(-cv_scores.mean()):.3f} (±{np.sqrt(cv_scores.std()):.3f})")
            print(f"   Test RMSE: {rmse:.3f}")
            print(f"   Test MAE: {mae:.3f}")
            print(f"   Test R²: {r2:.3f}")
        
        self.performance_metrics['regression'] = regression_results
        
        # Find best regression model
        best_regressor = min(regression_results.keys(), 
                           key=lambda x: regression_results[x]['test_rmse'])
        
        print(f"\\n🏆 Best Regression Model: {best_regressor.replace('_regressor', '').replace('_', ' ').title()}")
        print(f"   RMSE: {regression_results[best_regressor]['test_rmse']:.3f}")
        print(f"   R²: {regression_results[best_regressor]['test_r2']:.3f}")
        
        return regression_results
    
    def get_feature_importance(self):
        """Get feature importance from trained models"""
        
        importance_data = {}
        
        # XGBoost feature importance
        if 'xgboost' in self.models and hasattr(self.models['xgboost'], 'feature_importances_'):
            importance_data['xgboost'] = dict(zip(
                self.feature_names,
                [float(x) for x in self.models['xgboost'].feature_importances_]
            ))
        
        # Random Forest feature importance
        if 'random_forest' in self.models and hasattr(self.models['random_forest'], 'feature_importances_'):
            importance_data['random_forest'] = dict(zip(
                self.feature_names,
                [float(x) for x in self.models['random_forest'].feature_importances_]
            ))
        
        return importance_data
    
    def save_models(self, save_dir=None):
        """Save trained models and preprocessors"""
        
        if save_dir is None:
            save_dir = MODELS_DIR
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            if hasattr(model, 'predict'):  # Check if model is trained
                model_path = save_dir / f"{model_name}_model.joblib"
                joblib.dump(model, model_path)
                print(f"✓ Saved {model_name} to {model_path}")
        
        # Save preprocessors
        preprocessor_path = save_dir / "preprocessors.joblib"
        joblib.dump(self.preprocessors, preprocessor_path)
        print(f"✓ Saved preprocessors to {preprocessor_path}")
        
        # Save performance metrics
        metrics_path = save_dir / "performance_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        print(f"✓ Saved performance metrics to {metrics_path}")
        
        # Save feature importance
        importance = self.get_feature_importance()
        importance_path = save_dir / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=2)
        print(f"✓ Saved feature importance to {importance_path}")
    
    def predict_sleep_disorder(self, user_data, model_name='xgboost'):
        """Predict sleep disorder for new user data"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Convert single prediction to DataFrame
        if isinstance(user_data, dict):
            user_df = pd.DataFrame([user_data])
        else:
            user_df = user_data.copy()
        
        # Preprocess
        X_processed, _ = self.preprocess_data(user_df, target_col=None, fit_preprocessors=False)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probabilities = model.predict_proba(X_processed)[0]
        
        # Convert back to class names
        class_names = self.preprocessors['target_encoder'].classes_
        predicted_class = class_names[prediction]
        
        prob_dict = dict(zip(class_names, probabilities))
        
        return {
            'predicted_class': predicted_class,
            'probabilities': prob_dict,
            'confidence': max(probabilities)
        }
    
    def predict_sleep_quality(self, user_data, model_name='xgboost_regressor'):
        """Predict sleep quality for new user data"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Convert single prediction to DataFrame
        if isinstance(user_data, dict):
            user_df = pd.DataFrame([user_data])
        else:
            user_df = user_data.copy()
        
        # Preprocess (exclude Sleep Disorder and Quality of Sleep columns)
        if 'Sleep Disorder' in user_df.columns:
            user_df = user_df.drop('Sleep Disorder', axis=1)
        if 'Quality of Sleep' in user_df.columns:
            user_df = user_df.drop('Quality of Sleep', axis=1)
        
        X_processed, _ = self.preprocess_data(user_df, target_col=None, fit_preprocessors=False, use_regression_preprocessors=True)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        
        return {
            'predicted_quality': round(prediction, 2),
            'quality_level': 'Low' if prediction < 5 else 'Medium' if prediction < 7 else 'High'
        }


def main():
    """Main training pipeline"""
    
    print("🚀 Starting Sleep Health Ensemble Model Training")
    print("=" * 80)
    
    # Load cleaned data
    sleep_df = pd.read_csv(PROCESSED_DATA_DIR / 'sleep_health_cleaned.csv')
    print(f"✓ Loaded dataset: {sleep_df.shape}")
    
    # Initialize ensemble
    ensemble = SleepHealthEnsemble(random_state=42)
    
    # Train classification models
    classification_results = ensemble.train_classification_models(sleep_df)
    
    # Train regression models
    regression_results = ensemble.train_regression_models(sleep_df)
    
    # Print feature importance
    importance = ensemble.get_feature_importance()
    if importance:
        print("\\n📊 Feature Importance (XGBoost):")
        print("-" * 40)
        if 'xgboost' in importance:
            sorted_features = sorted(importance['xgboost'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, imp in sorted_features[:10]:
                print(f"   {feature:25} -> {imp:.3f}")
    
    # Save models
    print("\\n💾 Saving Models...")
    ensemble.save_models()
    
    print(f"\\n🎉 Training Complete!")
    print(f"Models saved to: {MODELS_DIR}")
    
    return ensemble, classification_results, regression_results


if __name__ == "__main__":
    ensemble, classification_results, regression_results = main()
