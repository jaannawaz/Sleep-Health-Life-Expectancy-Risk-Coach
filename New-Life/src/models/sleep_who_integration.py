"""
Sleep Health & WHO Integration Module
Combines ensemble sleep health models with WHO population context
for enriched, calibrated health predictions
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple, Union

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data'))

try:
    from config import *
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "src" / "models"

from ensemble_models import SleepHealthEnsemble
from who_integration import WHOHealthContext

class SleepHealthWHOPredictor:
    """
    Integrated Sleep Health Predictor with WHO Population Context
    
    Combines ensemble ML models for sleep health prediction with WHO population
    health data to provide calibrated, contextualized health assessments.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize integrated predictor with models and WHO context"""
        
        self.models_dir = models_dir or MODELS_DIR
        self.ensemble = SleepHealthEnsemble()
        self.who_context = WHOHealthContext()
        
        # Load trained models
        self._load_trained_models()
        
        # Default country for context
        self.default_country = 'United States of America'
        
        print("✓ Sleep Health & WHO Integration Ready")
    
    def _load_trained_models(self):
        """Load all trained ensemble models"""
        
        model_files = {
            'xgboost': 'xgboost_model.joblib',
            'random_forest': 'random_forest_model.joblib', 
            'logistic': 'logistic_model.joblib',
            'xgboost_regressor': 'xgboost_regressor_model.joblib',
            'rf_regressor': 'rf_regressor_model.joblib'
        }
        
        loaded_models = 0
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                self.ensemble.models[model_name] = joblib.load(model_path)
                loaded_models += 1
        
        # Load preprocessors
        preprocessor_path = self.models_dir / "preprocessors.joblib"
        if preprocessor_path.exists():
            self.ensemble.preprocessors = joblib.load(preprocessor_path)
            # Copy preprocessors for regression use
            self.ensemble.regression_preprocessors = self.ensemble.preprocessors.copy()
            
            # Set feature names for compatibility
            sample_features = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
                              'Physical Activity Level', 'Stress Level', 'BMI Category', 
                              'Blood Pressure', 'Heart Rate', 'Daily Steps']
            self.ensemble.feature_names = sample_features
            self.ensemble.regression_feature_names = [f for f in sample_features if f != 'Quality of Sleep']
            
        print(f"✓ Loaded {loaded_models} models and preprocessors")
    
    def predict_with_context(self, 
                           user_data: Dict, 
                           country: Optional[str] = None,
                           include_trends: bool = True,
                           model_preference: str = 'best') -> Dict:
        """
        Comprehensive sleep health prediction with WHO population context
        
        Args:
            user_data: Dictionary with user health and lifestyle data
            country: Country for population context (defaults to US)
            include_trends: Whether to include health trend analysis
            model_preference: 'best', 'xgboost', 'random_forest', or 'ensemble'
            
        Returns:
            Dictionary with predictions, context, and recommendations
        """
        
        country = country or self.default_country
        
        # Validate and prepare user data
        processed_user_data = self._prepare_user_data(user_data)
        
        # Get sleep disorder prediction
        disorder_prediction = self._get_sleep_disorder_prediction(
            processed_user_data, model_preference
        )
        
        # Get sleep quality prediction
        quality_prediction = self._get_sleep_quality_prediction(
            processed_user_data, model_preference
        )
        
        # Get WHO population context
        population_context = self.who_context.get_country_context(country)
        
        # Benchmark against population
        population_benchmark = self._benchmark_against_population(
            processed_user_data, country
        )
        
        # Calculate risk calibration
        risk_calibration = self._calibrate_risk_with_population(
            disorder_prediction, population_context, processed_user_data
        )
        
        # Get health trends if requested
        health_trends = None
        if include_trends and population_context:
            health_trends = self.who_context.get_health_trends(country, years=10)
        
        # Generate comprehensive recommendations
        recommendations = self._generate_recommendations(
            disorder_prediction, quality_prediction, population_benchmark, 
            risk_calibration, processed_user_data
        )
        
        # Compile comprehensive result
        result = {
            'user_profile': processed_user_data,
            'predictions': {
                'sleep_disorder': disorder_prediction,
                'sleep_quality': quality_prediction
            },
            'population_context': {
                'country': country,
                'context': population_context,
                'benchmark': population_benchmark,
                'risk_calibration': risk_calibration
            },
            'health_trends': health_trends,
            'recommendations': recommendations,
            'confidence_score': self._calculate_confidence_score(
                disorder_prediction, quality_prediction, population_context
            ),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return result
    
    def _prepare_user_data(self, user_data: Dict) -> Dict:
        """Prepare and validate user data for prediction"""
        
        processed = user_data.copy()
        
        # Store BMI numeric conversion separately for WHO integration
        # Don't add it to the main data as models weren't trained with it
        if 'BMI Category' in processed:
            bmi_mapping = {
                'Normal': 22.0,
                'Overweight': 27.5,
                'Obese': 32.5
            }
            processed['_bmi_numeric'] = bmi_mapping.get(processed['BMI Category'], 22.0)
        
        # Ensure required fields have defaults
        defaults = {
            'Gender': 'Male',
            'Age': 35,
            'Occupation': 'Teacher',
            'Sleep Duration': 7.0,
            'Quality of Sleep': 7,
            'Physical Activity Level': 50,
            'Stress Level': 5,
            'BMI Category': 'Normal',
            'Blood Pressure': '120/80',
            'Heart Rate': 70,
            'Daily Steps': 7000
        }
        
        for key, default_value in defaults.items():
            if key not in processed:
                processed[key] = default_value
        
        return processed
    
    def _get_sleep_disorder_prediction(self, user_data: Dict, model_preference: str) -> Dict:
        """Get sleep disorder prediction using specified model preference"""
        
        # Clean data for ensemble models (remove fields that weren't in training)
        clean_data = {k: v for k, v in user_data.items() if not k.startswith('_')}
        
        if model_preference == 'best':
            # Use Random Forest (best classifier from our results)
            model_name = 'random_forest'
        elif model_preference == 'ensemble':
            # Use voting from multiple models
            return self._get_ensemble_disorder_prediction(clean_data)
        else:
            model_name = model_preference
        
        if model_name in self.ensemble.models:
            return self.ensemble.predict_sleep_disorder(clean_data, model_name)
        else:
            # Fallback to random forest
            return self.ensemble.predict_sleep_disorder(clean_data, 'random_forest')
    
    def _get_sleep_quality_prediction(self, user_data: Dict, model_preference: str) -> Dict:
        """Get sleep quality prediction using specified model preference"""
        
        # Clean data for ensemble models (remove fields that weren't in training)
        clean_data = {k: v for k, v in user_data.items() if not k.startswith('_')}
        
        if model_preference == 'best':
            # Use XGBoost (best regressor from our results)
            model_name = 'xgboost_regressor'
        elif model_preference == 'ensemble':
            # Average of both regressors
            xgb_result = self.ensemble.predict_sleep_quality(clean_data, 'xgboost_regressor')
            rf_result = self.ensemble.predict_sleep_quality(clean_data, 'rf_regressor')
            
            avg_quality = (xgb_result['predicted_quality'] + rf_result['predicted_quality']) / 2
            return {
                'predicted_quality': round(avg_quality, 2),
                'quality_level': 'Low' if avg_quality < 5 else 'Medium' if avg_quality < 7 else 'High',
                'ensemble_method': 'averaged'
            }
        else:
            model_name = model_preference + '_regressor'
        
        if model_name in self.ensemble.models:
            return self.ensemble.predict_sleep_quality(clean_data, model_name)
        else:
            # Fallback to XGBoost regressor
            return self.ensemble.predict_sleep_quality(clean_data, 'xgboost_regressor')
    
    def _get_ensemble_disorder_prediction(self, user_data: Dict) -> Dict:
        """Get ensemble sleep disorder prediction from multiple models"""
        
        models_to_use = ['xgboost', 'random_forest', 'logistic']
        predictions = []
        probabilities = {}
        
        for model_name in models_to_use:
            if model_name in self.ensemble.models:
                result = self.ensemble.predict_sleep_disorder(user_data, model_name)
                predictions.append(result['predicted_class'])
                
                # Accumulate probabilities
                for disorder, prob in result['probabilities'].items():
                    if disorder not in probabilities:
                        probabilities[disorder] = []
                    probabilities[disorder].append(prob)
        
        # Voting for final prediction
        from collections import Counter
        vote_counts = Counter(predictions)
        final_prediction = vote_counts.most_common(1)[0][0]
        
        # Average probabilities
        avg_probabilities = {}
        for disorder, prob_list in probabilities.items():
            avg_probabilities[disorder] = sum(prob_list) / len(prob_list)
        
        return {
            'predicted_class': final_prediction,
            'probabilities': avg_probabilities,
            'confidence': max(avg_probabilities.values()),
            'ensemble_method': 'voting',
            'model_votes': dict(vote_counts)
        }
    
    def _benchmark_against_population(self, user_data: Dict, country: str) -> Dict:
        """Benchmark user data against country population"""
        
        # Prepare data for WHO benchmarking with proper field names
        who_data = user_data.copy()
        if '_bmi_numeric' in who_data:
            who_data['bmi_numeric'] = who_data.pop('_bmi_numeric')
        
        return self.who_context.benchmark_individual_against_population(
            who_data, country
        )
    
    def _calibrate_risk_with_population(self, 
                                      disorder_prediction: Dict, 
                                      population_context: Optional[Dict],
                                      user_data: Dict) -> Dict:
        """Calibrate individual risk using population health context"""
        
        if not population_context:
            return {'error': 'No population context available'}
        
        calibration = {
            'base_risk': disorder_prediction['confidence'],
            'population_adjusted_risk': disorder_prediction['confidence'],
            'adjustment_factors': {},
            'risk_level': 'Unknown'
        }
        
        # BMI-based adjustment
        user_bmi = user_data.get('_bmi_numeric', 22.0)
        pop_bmi = population_context['health_indicators'].get(' BMI ', user_bmi)
        
        if user_bmi > pop_bmi + 5:
            bmi_adjustment = 1.2  # Increase risk if significantly above population
            calibration['adjustment_factors']['BMI'] = 'Higher than population (+20%)'
        elif user_bmi > pop_bmi:
            bmi_adjustment = 1.1  # Slight increase
            calibration['adjustment_factors']['BMI'] = 'Above population (+10%)'
        else:
            bmi_adjustment = 1.0  # No adjustment
            calibration['adjustment_factors']['BMI'] = 'Within population range'
        
        # Age-based adjustment using life expectancy
        user_age = user_data.get('Age', 35)
        life_expectancy = population_context['health_indicators'].get('Life expectancy ', 75)
        age_ratio = user_age / life_expectancy
        
        if age_ratio > 0.6:  # Later life stage
            age_adjustment = 1.15
            calibration['adjustment_factors']['Age'] = 'Later life stage (+15%)'
        elif age_ratio > 0.4:  # Middle age
            age_adjustment = 1.05
            calibration['adjustment_factors']['Age'] = 'Middle age (+5%)'
        else:
            age_adjustment = 0.95
            calibration['adjustment_factors']['Age'] = 'Younger age (-5%)'
        
        # Socioeconomic adjustment
        income_score = population_context['health_indicators'].get('Income composition of resources', 0.7)
        if income_score < 0.5:
            socio_adjustment = 1.1
            calibration['adjustment_factors']['Socioeconomic'] = 'Lower income context (+10%)'
        else:
            socio_adjustment = 1.0
            calibration['adjustment_factors']['Socioeconomic'] = 'Adequate income context'
        
        # Calculate adjusted risk
        total_adjustment = bmi_adjustment * age_adjustment * socio_adjustment
        calibration['population_adjusted_risk'] = min(
            calibration['base_risk'] * total_adjustment, 1.0
        )
        
        # Determine risk level
        adjusted_risk = calibration['population_adjusted_risk']
        if adjusted_risk > 0.8:
            calibration['risk_level'] = 'Very High'
        elif adjusted_risk > 0.6:
            calibration['risk_level'] = 'High'
        elif adjusted_risk > 0.4:
            calibration['risk_level'] = 'Moderate'
        else:
            calibration['risk_level'] = 'Low'
        
        return calibration
    
    def _generate_recommendations(self, 
                                disorder_prediction: Dict,
                                quality_prediction: Dict,
                                population_benchmark: Dict,
                                risk_calibration: Dict,
                                user_data: Dict) -> List[Dict]:
        """Generate comprehensive health recommendations"""
        
        recommendations = []
        
        # Disorder-specific recommendations
        predicted_disorder = disorder_prediction['predicted_class']
        if predicted_disorder != 'None':
            if predicted_disorder == 'Sleep Apnea':
                recommendations.append({
                    'category': 'Medical',
                    'priority': 'High',
                    'recommendation': 'Consult a sleep specialist for sleep apnea evaluation and possible CPAP therapy',
                    'reason': f'High probability of sleep apnea ({disorder_prediction["confidence"]:.1%})'
                })
                recommendations.append({
                    'category': 'Lifestyle',
                    'priority': 'High',
                    'recommendation': 'Consider weight management and sleep position changes',
                    'reason': 'Sleep apnea is often associated with weight and sleep position'
                })
            
            elif predicted_disorder == 'Insomnia':
                recommendations.append({
                    'category': 'Behavioral',
                    'priority': 'High',
                    'recommendation': 'Implement sleep hygiene practices and consider cognitive behavioral therapy for insomnia (CBT-I)',
                    'reason': f'High probability of insomnia ({disorder_prediction["confidence"]:.1%})'
                })
                recommendations.append({
                    'category': 'Lifestyle',
                    'priority': 'Medium',
                    'recommendation': 'Establish consistent sleep schedule and limit screen time before bed',
                    'reason': 'Consistent routines help regulate circadian rhythm'
                })
        
        # Sleep quality recommendations
        sleep_quality = quality_prediction['predicted_quality']
        if sleep_quality < 6:
            recommendations.append({
                'category': 'Sleep Quality',
                'priority': 'Medium',
                'recommendation': 'Focus on sleep environment optimization (temperature, darkness, noise control)',
                'reason': f'Predicted sleep quality is below average ({sleep_quality:.1f}/10)'
            })
        
        # BMI-based recommendations
        if 'individual_vs_population' in population_benchmark:
            bmi_data = population_benchmark['individual_vs_population'].get('BMI', {})
            if bmi_data.get('relative_risk') == 'Higher':
                recommendations.append({
                    'category': 'Lifestyle',
                    'priority': 'Medium',
                    'recommendation': 'Consider gradual weight management through diet and exercise',
                    'reason': f'BMI is above population average in your country context'
                })
        
        # Risk calibration recommendations
        risk_level = risk_calibration.get('risk_level', 'Unknown')
        if risk_level in ['High', 'Very High']:
            recommendations.append({
                'category': 'Preventive',
                'priority': 'High',
                'recommendation': 'Schedule comprehensive health checkup and discuss sleep concerns with healthcare provider',
                'reason': f'Population-adjusted risk level is {risk_level.lower()}'
            })
        
        # Activity and stress recommendations
        activity_level = user_data.get('Physical Activity Level', 50)
        stress_level = user_data.get('Stress Level', 5)
        
        if activity_level < 40:
            recommendations.append({
                'category': 'Lifestyle',
                'priority': 'Medium',
                'recommendation': 'Gradually increase physical activity to 150 minutes of moderate exercise per week',
                'reason': 'Low physical activity can negatively impact sleep quality'
            })
        
        if stress_level > 6:
            recommendations.append({
                'category': 'Mental Health',
                'priority': 'Medium',
                'recommendation': 'Explore stress management techniques such as meditation, yoga, or counseling',
                'reason': 'High stress levels are strongly associated with sleep disorders'
            })
        
        # Sleep duration recommendations
        sleep_duration = user_data.get('Sleep Duration', 7)
        if sleep_duration < 6.5:
            recommendations.append({
                'category': 'Sleep Hygiene',
                'priority': 'High',
                'recommendation': 'Aim for 7-9 hours of sleep per night by adjusting bedtime routine',
                'reason': 'Short sleep duration significantly increases sleep disorder risk'
            })
        elif sleep_duration > 9:
            recommendations.append({
                'category': 'Sleep Hygiene',
                'priority': 'Medium',
                'recommendation': 'Evaluate reasons for excessive sleep and consider sleep study if persistent',
                'reason': 'Excessive sleep may indicate underlying health issues'
            })
        
        return recommendations
    
    def _calculate_confidence_score(self, 
                                  disorder_prediction: Dict,
                                  quality_prediction: Dict,
                                  population_context: Optional[Dict]) -> float:
        """Calculate overall confidence score for the prediction"""
        
        # Base confidence from model predictions
        disorder_confidence = disorder_prediction.get('confidence', 0.5)
        
        # Quality prediction doesn't have direct confidence, so estimate based on consistency
        # This is a simplified approach - in practice, you'd want prediction intervals
        quality_confidence = 0.8  # Assume high confidence for regression
        
        # Population context availability bonus
        context_bonus = 0.1 if population_context else 0.0
        
        # Average the confidences and add context bonus
        overall_confidence = (disorder_confidence + quality_confidence) / 2 + context_bonus
        
        return min(overall_confidence, 1.0)
    
    def get_available_countries(self) -> List[str]:
        """Get list of countries available for population context"""
        return self.who_context.get_available_countries()
    
    def get_major_countries(self) -> List[str]:
        """Get list of major countries with complete data"""
        return self.who_context.get_major_countries()
    
    def compare_user_across_countries(self, 
                                    user_data: Dict, 
                                    countries: List[str]) -> Dict:
        """Compare user risk across different countries"""
        
        comparisons = {}
        
        for country in countries:
            try:
                prediction = self.predict_with_context(
                    user_data, country=country, include_trends=False
                )
                
                comparisons[country] = {
                    'disorder_risk': prediction['predictions']['sleep_disorder']['confidence'],
                    'quality_score': prediction['predictions']['sleep_quality']['predicted_quality'],
                    'risk_level': prediction['population_context']['risk_calibration']['risk_level'],
                    'adjusted_risk': prediction['population_context']['risk_calibration']['population_adjusted_risk']
                }
                
            except Exception as e:
                comparisons[country] = {'error': str(e)}
        
        return {
            'user_profile': user_data,
            'country_comparisons': comparisons,
            'timestamp': pd.Timestamp.now().isoformat()
        }


def main():
    """Test integrated sleep health and WHO prediction system"""
    
    print("🔄 Testing Sleep Health & WHO Integration")
    print("=" * 60)
    
    # Initialize integrated predictor
    predictor = SleepHealthWHOPredictor()
    
    # Test with sample user data
    test_user = {
        'Gender': 'Male',
        'Age': 42,
        'Occupation': 'Software Engineer',
        'Sleep Duration': 6.0,
        'Quality of Sleep': 5,
        'Physical Activity Level': 35,
        'Stress Level': 7,
        'BMI Category': 'Overweight',
        'Blood Pressure': '135/85',
        'Heart Rate': 78,
        'Daily Steps': 4500
    }
    
    print(f"👤 Test User Profile:")
    for key, value in test_user.items():
        print(f"   {key:25} -> {value}")
    
    # Get comprehensive prediction
    print(f"\\n🔮 Getting Comprehensive Prediction...")
    result = predictor.predict_with_context(
        test_user, 
        country='United States of America',
        model_preference='best'
    )
    
    # Display results
    print(f"\\n📊 PREDICTION RESULTS:")
    print("-" * 40)
    
    # Sleep disorder prediction
    disorder = result['predictions']['sleep_disorder']
    print(f"Sleep Disorder Risk: {disorder['predicted_class']}")
    print(f"Confidence: {disorder['confidence']:.1%}")
    
    # Sleep quality prediction
    quality = result['predictions']['sleep_quality']
    print(f"Sleep Quality: {quality['predicted_quality']:.1f}/10 ({quality['quality_level']})")
    
    # Population context
    pop_context = result['population_context']
    print(f"\\n🌍 POPULATION CONTEXT:")
    print("-" * 40)
    print(f"Country: {pop_context['country']}")
    
    risk_cal = pop_context['risk_calibration']
    print(f"Base Risk: {risk_cal['base_risk']:.1%}")
    print(f"Population-Adjusted Risk: {risk_cal['population_adjusted_risk']:.1%}")
    print(f"Risk Level: {risk_cal['risk_level']}")
    
    # Recommendations
    recommendations = result['recommendations']
    print(f"\\n💡 RECOMMENDATIONS ({len(recommendations)} items):")
    print("-" * 40)
    
    for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
        print(f"{i}. [{rec['category']}] {rec['recommendation']}")
        print(f"   Priority: {rec['priority']} | Reason: {rec['reason']}")
        print()
    
    print(f"Overall Confidence: {result['confidence_score']:.1%}")
    
    # Test country comparison
    print(f"\\n🌎 COUNTRY COMPARISON TEST:")
    print("-" * 40)
    
    major_countries = predictor.get_major_countries()[:3]
    comparison = predictor.compare_user_across_countries(test_user, major_countries)
    
    for country, data in comparison['country_comparisons'].items():
        if 'error' not in data:
            print(f"{country}:")
            print(f"  Risk Level: {data['risk_level']}")
            print(f"  Adjusted Risk: {data['adjusted_risk']:.1%}")
            print(f"  Quality Score: {data['quality_score']:.1f}")
        print()
    
    print("✅ Integration testing complete!")
    
    return predictor, result


if __name__ == "__main__":
    predictor, result = main()
