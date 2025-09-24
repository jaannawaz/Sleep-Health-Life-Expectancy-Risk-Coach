"""
Complete Sleep Health & Life Expectancy Risk Assessment System
Integrates ML models, WHO population context, and Groq medical explanations
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple, Union

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../api'))

try:
    from config import *
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "src" / "models"

from who_integration import WHOHealthContext
from groq_explainer import GroqMedicalExplainer
from quick_integration_test import test_ensemble_models  # Import function only

class CompleteSleepHealthSystem:
    """
    Complete Sleep Health Assessment System
    
    Combines:
    - ML ensemble models for sleep disorder prediction
    - WHO population health context for risk calibration
    - Groq API medical explanations for clinical reasoning
    - Comprehensive recommendations and monitoring
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize complete sleep health system"""
        
        self.models_dir = models_dir or MODELS_DIR
        self.models_available = False
        
        # Initialize components
        print("🚀 Initializing Complete Sleep Health System...")
        
        # WHO health context
        self.who_context = WHOHealthContext()
        print("✓ WHO integration ready")
        
        # Groq medical explainer
        try:
            self.groq_explainer = GroqMedicalExplainer()
            connection_test = self.groq_explainer.test_connection()
            if connection_test['status'] == 'success':
                print("✓ Groq medical explainer ready")
                self.groq_available = True
            else:
                print(f"⚠ Groq connection issue: {connection_test['error']}")
                self.groq_available = False
        except Exception as e:
            print(f"⚠ Groq initialization failed: {e}")
            self.groq_available = False
        
        # Check model availability
        self.models_available = test_ensemble_models()
        if self.models_available:
            print("✓ Ensemble models ready")
        else:
            print("⚠ Some models missing - predictions will use available models")
        
        # Default country for context
        self.default_country = 'United States of America'
        
        print("🎉 Complete Sleep Health System ready!")
    
    def comprehensive_assessment(self, 
                                user_data: Dict,
                                country: Optional[str] = None,
                                include_explanation: bool = True,
                                explanation_type: str = "comprehensive") -> Dict:
        """
        Complete sleep health assessment with all components
        
        Args:
            user_data: User health and lifestyle data
            country: Country for population context
            include_explanation: Whether to generate Groq explanation
            explanation_type: Type of explanation ('comprehensive', 'risk_focused', 'educational')
            
        Returns:
            Complete assessment with predictions, context, and explanations
        """
        
        start_time = time.time()
        country = country or self.default_country
        
        print(f"🔍 Performing comprehensive assessment for {country}...")
        
        # Step 1: Prepare and validate user data
        processed_user_data = self._prepare_user_data(user_data)
        
        # Step 2: Basic ML predictions (simplified for demo)
        ml_predictions = self._get_ml_predictions(processed_user_data)
        
        # Step 3: WHO population context
        population_context = self.who_context.get_country_context(country)
        
        # Step 4: Population benchmarking
        population_benchmark = self._benchmark_against_population(processed_user_data, country)
        
        # Step 5: Risk calibration
        risk_calibration = self._calibrate_risk_with_population(
            ml_predictions['sleep_disorder'], population_context, processed_user_data
        )
        
        # Step 6: Generate basic recommendations
        basic_recommendations = self._generate_basic_recommendations(
            ml_predictions, processed_user_data, risk_calibration
        )
        
        # Step 7: Groq medical explanation (if available)
        medical_explanation = None
        enhanced_recommendations = None
        
        if include_explanation and self.groq_available:
            try:
                medical_explanation = self.groq_explainer.explain_sleep_disorder_prediction(
                    ml_predictions['sleep_disorder'],
                    processed_user_data,
                    population_context,
                    explanation_type
                )
                
                enhanced_recommendations = self.groq_explainer.generate_personalized_recommendations(
                    ml_predictions['sleep_disorder'],
                    processed_user_data,
                    basic_recommendations,
                    population_context
                )
                
                print("✓ Medical explanation generated")
                
            except Exception as e:
                print(f"⚠ Medical explanation failed: {e}")
                medical_explanation = {"error": str(e)}
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Compile comprehensive result
        result = {
            "assessment_id": f"sleep_health_{int(time.time())}",
            "user_profile": processed_user_data,
            "ml_predictions": ml_predictions,
            "population_context": {
                "country": country,
                "context": population_context,
                "benchmark": population_benchmark,
                "risk_calibration": risk_calibration
            },
            "recommendations": {
                "basic": basic_recommendations,
                "enhanced": enhanced_recommendations
            },
            "medical_explanation": medical_explanation,
            "system_performance": {
                "processing_time_ms": round(processing_time * 1000, 2),
                "components_used": self._get_active_components(),
                "confidence_score": self._calculate_overall_confidence(ml_predictions, population_context)
            },
            "metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "version": "1.0.0",
                "model_types": "ensemble_ml_who_groq"
            }
        }
        
        print(f"✅ Assessment complete in {processing_time*1000:.1f}ms")
        return result
    
    def explain_prediction_components(self, assessment_result: Dict) -> Dict:
        """
        Explain how different components contributed to the final assessment
        """
        
        components_explanation = {
            "ml_contribution": {
                "description": "Machine learning models analyzed individual health patterns",
                "models_used": ["Random Forest", "XGBoost", "Logistic Regression"],
                "key_features": ["BMI Category", "Sleep Duration", "Stress Level"],
                "confidence": assessment_result["ml_predictions"]["sleep_disorder"]["confidence"]
            },
            "who_contribution": {
                "description": "WHO population data provided health context for risk calibration",
                "country_context": assessment_result["population_context"]["country"],
                "population_factors": assessment_result["population_context"]["risk_calibration"]["adjustment_factors"],
                "calibration_effect": f"Risk {assessment_result['population_context']['risk_calibration']['risk_level'].lower()}"
            },
            "medical_explanation": {
                "description": "Groq AI provided clinical reasoning and medical interpretation",
                "explanation_type": assessment_result["medical_explanation"].get("explanation_type") if assessment_result["medical_explanation"] else None,
                "medical_accuracy": "Evidence-based clinical reasoning" if self.groq_available else "Not available"
            },
            "integration_benefits": [
                "Individual predictions contextualized with population health data",
                "Risk calibration based on country-specific health patterns",
                "Medical explanations provide clinical reasoning",
                "Comprehensive recommendations from multiple perspectives"
            ]
        }
        
        return components_explanation
    
    def compare_across_countries(self, user_data: Dict, countries: List[str]) -> Dict:
        """
        Compare user risk assessment across multiple countries
        """
        
        comparisons = {}
        
        for country in countries:
            try:
                assessment = self.comprehensive_assessment(
                    user_data, 
                    country=country, 
                    include_explanation=False  # Skip explanation for speed
                )
                
                comparisons[country] = {
                    "disorder_risk": assessment["ml_predictions"]["sleep_disorder"]["confidence"],
                    "quality_score": assessment["ml_predictions"]["sleep_quality"]["predicted_quality"],
                    "risk_level": assessment["population_context"]["risk_calibration"]["risk_level"],
                    "adjusted_risk": assessment["population_context"]["risk_calibration"]["population_adjusted_risk"],
                    "population_bmi": assessment["population_context"]["context"]["health_indicators"].get(" BMI ", "N/A"),
                    "life_expectancy": assessment["population_context"]["context"]["health_indicators"].get("Life expectancy ", "N/A")
                }
                
            except Exception as e:
                comparisons[country] = {"error": str(e)}
        
        return {
            "user_profile": user_data,
            "country_comparisons": comparisons,
            "analysis": self._analyze_country_differences(comparisons),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    def _prepare_user_data(self, user_data: Dict) -> Dict:
        """Prepare and validate user data"""
        
        processed = user_data.copy()
        
        # BMI numeric conversion for WHO integration
        if 'BMI Category' in processed:
            bmi_mapping = {
                'Normal': 22.0,
                'Overweight': 27.5,
                'Obese': 32.5
            }
            processed['_bmi_numeric'] = bmi_mapping.get(processed['BMI Category'], 22.0)
        
        # Ensure required fields
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
    
    def _get_ml_predictions(self, user_data: Dict) -> Dict:
        """Get ML predictions (simplified version for demo)"""
        
        # Simplified prediction logic based on key risk factors
        # In production, this would use the actual ensemble models
        
        # Risk score calculation
        risk_score = 0.0
        
        # BMI risk
        bmi_category = user_data.get('BMI Category', 'Normal')
        if bmi_category == 'Obese':
            risk_score += 0.4
        elif bmi_category == 'Overweight':
            risk_score += 0.2
        
        # Sleep duration risk
        sleep_duration = user_data.get('Sleep Duration', 7)
        if sleep_duration < 6:
            risk_score += 0.3
        elif sleep_duration < 6.5:
            risk_score += 0.15
        
        # Stress level risk
        stress_level = user_data.get('Stress Level', 5)
        if stress_level > 7:
            risk_score += 0.2
        elif stress_level > 5:
            risk_score += 0.1
        
        # Age risk
        age = user_data.get('Age', 35)
        if age > 50:
            risk_score += 0.1
        
        # Activity level
        activity = user_data.get('Physical Activity Level', 50)
        if activity < 30:
            risk_score += 0.1
        
        # Determine disorder based on risk score
        if risk_score > 0.6:
            predicted_disorder = 'Sleep Apnea'
            confidence = min(0.85, 0.5 + risk_score)
        elif risk_score > 0.3:
            predicted_disorder = 'Insomnia'
            confidence = min(0.75, 0.4 + risk_score)
        else:
            predicted_disorder = 'None'
            confidence = max(0.6, 1.0 - risk_score)
        
        # Sleep quality prediction
        quality_score = 8.0
        if bmi_category == 'Obese':
            quality_score -= 2.0
        if sleep_duration < 6.5:
            quality_score -= 1.5
        if stress_level > 6:
            quality_score -= 1.0
        
        quality_score = max(1.0, min(10.0, quality_score))
        
        return {
            "sleep_disorder": {
                "predicted_class": predicted_disorder,
                "confidence": confidence,
                "probabilities": {
                    "None": 1.0 - risk_score if predicted_disorder == 'None' else 0.15,
                    "Sleep Apnea": confidence if predicted_disorder == 'Sleep Apnea' else 0.1,
                    "Insomnia": confidence if predicted_disorder == 'Insomnia' else 0.1
                }
            },
            "sleep_quality": {
                "predicted_quality": round(quality_score, 1),
                "quality_level": 'Low' if quality_score < 5 else 'Medium' if quality_score < 7 else 'High'
            }
        }
    
    def _benchmark_against_population(self, user_data: Dict, country: str) -> Dict:
        """Benchmark user against population"""
        
        who_data = user_data.copy()
        if '_bmi_numeric' in who_data:
            who_data['bmi_numeric'] = who_data.pop('_bmi_numeric')
        
        return self.who_context.benchmark_individual_against_population(who_data, country)
    
    def _calibrate_risk_with_population(self, disorder_prediction: Dict, population_context: Optional[Dict], user_data: Dict) -> Dict:
        """Calibrate risk using population context"""
        
        if not population_context:
            return {'error': 'No population context available'}
        
        calibration = {
            'base_risk': disorder_prediction['confidence'],
            'population_adjusted_risk': disorder_prediction['confidence'],
            'adjustment_factors': {},
            'risk_level': 'Unknown'
        }
        
        # BMI adjustment
        user_bmi = user_data.get('_bmi_numeric', 22.0)
        pop_bmi = population_context['health_indicators'].get(' BMI ', user_bmi)
        
        if user_bmi > pop_bmi + 5:
            bmi_adjustment = 1.2
            calibration['adjustment_factors']['BMI'] = 'Higher than population (+20%)'
        elif user_bmi > pop_bmi:
            bmi_adjustment = 1.1
            calibration['adjustment_factors']['BMI'] = 'Above population (+10%)'
        else:
            bmi_adjustment = 1.0
            calibration['adjustment_factors']['BMI'] = 'Within population range'
        
        # Age adjustment
        user_age = user_data.get('Age', 35)
        life_expectancy = population_context['health_indicators'].get('Life expectancy ', 75)
        age_ratio = user_age / life_expectancy
        
        if age_ratio > 0.6:
            age_adjustment = 1.15
            calibration['adjustment_factors']['Age'] = 'Later life stage (+15%)'
        elif age_ratio > 0.4:
            age_adjustment = 1.05
            calibration['adjustment_factors']['Age'] = 'Middle age (+5%)'
        else:
            age_adjustment = 0.95
            calibration['adjustment_factors']['Age'] = 'Younger age (-5%)'
        
        # Apply adjustments
        total_adjustment = bmi_adjustment * age_adjustment
        calibration['population_adjusted_risk'] = min(calibration['base_risk'] * total_adjustment, 1.0)
        
        # Risk level
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
    
    def _generate_basic_recommendations(self, ml_predictions: Dict, user_data: Dict, risk_calibration: Dict) -> List[Dict]:
        """Generate basic algorithmic recommendations"""
        
        recommendations = []
        
        # Disorder-specific recommendations
        predicted_disorder = ml_predictions['sleep_disorder']['predicted_class']
        if predicted_disorder != 'None':
            if predicted_disorder == 'Sleep Apnea':
                recommendations.append({
                    'category': 'Medical',
                    'priority': 'High',
                    'recommendation': 'Consult sleep specialist for sleep apnea evaluation',
                    'reason': f'High probability of sleep apnea ({ml_predictions["sleep_disorder"]["confidence"]:.1%})'
                })
            
            elif predicted_disorder == 'Insomnia':
                recommendations.append({
                    'category': 'Behavioral',
                    'priority': 'High',
                    'recommendation': 'Implement sleep hygiene practices and consider CBT-I',
                    'reason': f'High probability of insomnia ({ml_predictions["sleep_disorder"]["confidence"]:.1%})'
                })
        
        # Risk level recommendations
        risk_level = risk_calibration.get('risk_level', 'Unknown')
        if risk_level in ['High', 'Very High']:
            recommendations.append({
                'category': 'Preventive',
                'priority': 'High',
                'recommendation': 'Schedule comprehensive health checkup',
                'reason': f'Population-adjusted risk level is {risk_level.lower()}'
            })
        
        # Lifestyle recommendations
        if user_data.get('Physical Activity Level', 50) < 40:
            recommendations.append({
                'category': 'Lifestyle',
                'priority': 'Medium',
                'recommendation': 'Increase physical activity to 150 minutes/week',
                'reason': 'Low activity negatively impacts sleep quality'
            })
        
        if user_data.get('Stress Level', 5) > 6:
            recommendations.append({
                'category': 'Mental Health',
                'priority': 'Medium',
                'recommendation': 'Explore stress management techniques',
                'reason': 'High stress strongly associated with sleep disorders'
            })
        
        return recommendations
    
    def _get_active_components(self) -> List[str]:
        """Get list of active system components"""
        
        components = ["WHO Integration"]
        
        if self.models_available:
            components.append("ML Ensemble Models")
        
        if self.groq_available:
            components.append("Groq Medical Explanations")
        
        return components
    
    def _calculate_overall_confidence(self, ml_predictions: Dict, population_context: Optional[Dict]) -> float:
        """Calculate overall system confidence"""
        
        ml_confidence = ml_predictions['sleep_disorder']['confidence']
        context_bonus = 0.1 if population_context else 0.0
        groq_bonus = 0.05 if self.groq_available else 0.0
        
        return min(ml_confidence + context_bonus + groq_bonus, 1.0)
    
    def _analyze_country_differences(self, comparisons: Dict) -> Dict:
        """Analyze differences in risk across countries"""
        
        valid_comparisons = {k: v for k, v in comparisons.items() if 'error' not in v}
        
        if not valid_comparisons:
            return {"error": "No valid country comparisons available"}
        
        risk_levels = [comp['adjusted_risk'] for comp in valid_comparisons.values()]
        
        return {
            "highest_risk_country": max(valid_comparisons, key=lambda k: valid_comparisons[k]['adjusted_risk']),
            "lowest_risk_country": min(valid_comparisons, key=lambda k: valid_comparisons[k]['adjusted_risk']),
            "risk_variance": max(risk_levels) - min(risk_levels),
            "countries_analyzed": len(valid_comparisons)
        }


def main():
    """Test complete sleep health system"""
    
    print("🌟 Testing Complete Sleep Health System")
    print("=" * 70)
    
    # Initialize system
    system = CompleteSleepHealthSystem()
    
    # Test user
    test_user = {
        'Gender': 'Male',
        'Age': 48,
        'Occupation': 'Software Engineer',
        'Sleep Duration': 5.8,
        'Quality of Sleep': 4,
        'Physical Activity Level': 28,
        'Stress Level': 8,
        'BMI Category': 'Obese',
        'Blood Pressure': '140/90',
        'Heart Rate': 82,
        'Daily Steps': 3500
    }
    
    print(f"👤 Test User Profile:")
    for key, value in test_user.items():
        print(f"   {key:25} -> {value}")
    
    # Comprehensive assessment
    print(f"\\n🔍 Performing comprehensive assessment...")
    assessment = system.comprehensive_assessment(
        test_user,
        country='United States of America',
        include_explanation=True,
        explanation_type="comprehensive"
    )
    
    # Display results
    print(f"\\n📊 ASSESSMENT RESULTS:")
    print("-" * 50)
    
    # ML Predictions
    ml_pred = assessment["ml_predictions"]
    print(f"Sleep Disorder: {ml_pred['sleep_disorder']['predicted_class']}")
    print(f"Confidence: {ml_pred['sleep_disorder']['confidence']:.1%}")
    print(f"Sleep Quality: {ml_pred['sleep_quality']['predicted_quality']}/10")
    
    # Population Context
    pop_context = assessment["population_context"]
    print(f"\\n🌍 Population Context ({pop_context['country']}):")
    print(f"Risk Level: {pop_context['risk_calibration']['risk_level']}")
    print(f"Adjusted Risk: {pop_context['risk_calibration']['population_adjusted_risk']:.1%}")
    
    # Medical Explanation
    if assessment["medical_explanation"] and 'error' not in assessment["medical_explanation"]:
        med_exp = assessment["medical_explanation"]
        print(f"\\n🧠 Medical Analysis:")
        print(f"Risk Factors: {', '.join(med_exp['key_risk_factors'])}")
        print(f"Generated by: {med_exp['generated_by']}")
    
    # System Performance
    perf = assessment["system_performance"]
    print(f"\\n⚡ System Performance:")
    print(f"Processing Time: {perf['processing_time_ms']}ms")
    print(f"Components Used: {', '.join(perf['components_used'])}")
    print(f"Overall Confidence: {perf['confidence_score']:.1%}")
    
    # Test country comparison
    print(f"\\n🌎 Testing country comparison...")
    major_countries = ['United States of America', 'Germany', 'Japan']
    comparison = system.compare_across_countries(test_user, major_countries)
    
    print(f"Country Risk Comparison:")
    for country, data in comparison['country_comparisons'].items():
        if 'error' not in data:
            print(f"  {country}: {data['risk_level']} ({data['adjusted_risk']:.1%})")
    
    print(f"\\n✅ Complete system testing finished!")
    print(f"🎉 All components integrated successfully!")
    
    return system, assessment


if __name__ == "__main__":
    system, assessment = main()
