"""
Groq API Integration for Medical Sleep Health Explanations
Provides clinical-style reasoning and recommendations using fast LLM inference
"""

import json
import time
from typing import Dict, List, Optional, Union
from groq import Groq
import sys
import os
from pathlib import Path

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))
try:
    from config import API_CONFIG
except ImportError:
    # Fallback configuration
    API_CONFIG = {
        "groq": {
            "api_key": os.getenv("GROQ_API_KEY", ""),
            "model": "openai/gpt-oss-120b",
            "temperature": 0.3,
            "max_completion_tokens": 2048,
            "timeout": 30,
            "max_retries": 3
        }
    }

class GroqMedicalExplainer:
    """
    Medical explanation service using Groq's fast LLM inference
    
    Provides clinical-style reasoning, risk factor analysis, and personalized
    recommendations for sleep health predictions with population context.
    """
    
    def __init__(self):
        """Initialize Groq client with configuration"""
        
        self.config = API_CONFIG["groq"]
        self.client = Groq(api_key=self.config["api_key"])
        self.model = self.config["model"]
        
        # Medical prompt templates
        self.system_prompts = {
            "sleep_specialist": """You are an experienced sleep medicine specialist providing clinical explanations. 
            Your responses should be:
            - Medically accurate and evidence-based
            - Clear for both patients and healthcare providers
            - Include specific recommendations
            - Reference population health context when provided
            - Maintain professional medical tone
            - Always include appropriate disclaimers about seeking professional medical advice""",
            
            "risk_analyzer": """You are a medical risk assessment specialist. Analyze sleep health predictions and provide:
            - Clear risk factor identification
            - Population-adjusted risk interpretation
            - Evidence-based lifestyle recommendations
            - When to seek medical attention
            - Preventive measures based on risk level""",
            
            "health_educator": """You are a health educator specializing in sleep medicine. Provide:
            - Easy-to-understand explanations of sleep health
            - Actionable lifestyle recommendations
            - Educational content about sleep disorders
            - Motivation for healthy sleep habits
            - Population health context interpretation"""
        }
        
        print("✓ Groq Medical Explainer initialized")
    
    def explain_sleep_disorder_prediction(self, 
                                        prediction_result: Dict,
                                        user_data: Dict,
                                        population_context: Optional[Dict] = None,
                                        explanation_type: str = "comprehensive") -> Dict:
        """
        Generate medical explanation for sleep disorder prediction
        
        Args:
            prediction_result: ML model prediction results
            user_data: Individual user health and lifestyle data
            population_context: WHO population health context
            explanation_type: Type of explanation ('comprehensive', 'risk_focused', 'educational')
            
        Returns:
            Dictionary with medical explanation and recommendations
        """
        
        # Choose appropriate system prompt
        if explanation_type == "risk_focused":
            system_prompt = self.system_prompts["risk_analyzer"]
        elif explanation_type == "educational":
            system_prompt = self.system_prompts["health_educator"]
        else:
            system_prompt = self.system_prompts["sleep_specialist"]
        
        # Build comprehensive context
        context = self._build_medical_context(prediction_result, user_data, population_context)
        
        # Create medical explanation prompt
        user_prompt = f"""
        Please provide a comprehensive medical explanation for this sleep health assessment:

        PREDICTION RESULTS:
        - Predicted Sleep Disorder: {prediction_result.get('predicted_class', 'Unknown')}
        - Confidence Level: {prediction_result.get('confidence', 0)*100:.1f}%
        - Risk Probabilities: {json.dumps(prediction_result.get('probabilities', {}), indent=2)}

        PATIENT PROFILE:
        - Age: {user_data.get('Age', 'Unknown')} years
        - Gender: {user_data.get('Gender', 'Unknown')}
        - BMI Category: {user_data.get('BMI Category', 'Unknown')}
        - Sleep Duration: {user_data.get('Sleep Duration', 'Unknown')} hours
        - Sleep Quality: {user_data.get('Quality of Sleep', 'Unknown')}/10
        - Physical Activity Level: {user_data.get('Physical Activity Level', 'Unknown')}
        - Stress Level: {user_data.get('Stress Level', 'Unknown')}/10
        - Occupation: {user_data.get('Occupation', 'Unknown')}
        - Heart Rate: {user_data.get('Heart Rate', 'Unknown')} bpm
        - Daily Steps: {user_data.get('Daily Steps', 'Unknown')}

        {context}

        Please provide:
        1. MEDICAL INTERPRETATION: What do these results mean clinically?
        2. RISK FACTOR ANALYSIS: Which factors are most concerning and why?
        3. POPULATION CONTEXT: How does this compare to population health patterns?
        4. RECOMMENDATIONS: Specific, actionable medical and lifestyle advice
        5. NEXT STEPS: When to seek professional medical attention
        6. DISCLAIMER: Appropriate medical disclaimer

        Format as clear, structured medical explanation suitable for patient education.
        """
        
        try:
            # Generate explanation using Groq
            explanation = self._generate_explanation(system_prompt, user_prompt)
            
            # Parse and structure the response
            structured_explanation = self._structure_explanation(explanation, prediction_result, user_data)
            
            return structured_explanation
            
        except Exception as e:
            return {
                "error": f"Failed to generate medical explanation: {str(e)}",
                "fallback_explanation": self._generate_fallback_explanation(prediction_result, user_data),
                "timestamp": time.time()
            }
    
    def explain_risk_calibration(self,
                               base_prediction: Dict,
                               risk_calibration: Dict,
                               population_context: Dict) -> Dict:
        """
        Explain population-adjusted risk calibration in medical terms
        
        Args:
            base_prediction: Original ML prediction
            risk_calibration: Population-adjusted risk results
            population_context: WHO population health data
            
        Returns:
            Medical explanation of risk calibration
        """
        
        user_prompt = f"""
        Please explain this population-adjusted sleep health risk assessment:

        ORIGINAL PREDICTION:
        - Base Risk: {base_prediction.get('confidence', 0)*100:.1f}%
        - Predicted Disorder: {base_prediction.get('predicted_class', 'Unknown')}

        POPULATION-ADJUSTED ASSESSMENT:
        - Adjusted Risk: {risk_calibration.get('population_adjusted_risk', 0)*100:.1f}%
        - Risk Level: {risk_calibration.get('risk_level', 'Unknown')}
        - Adjustment Factors: {json.dumps(risk_calibration.get('adjustment_factors', {}), indent=2)}

        POPULATION CONTEXT:
        - Country: {population_context.get('country', 'Unknown')}
        - Development Status: {population_context.get('status', 'Unknown')}
        - Population Health Indicators: {json.dumps(population_context.get('health_indicators', {}), indent=2)}

        Please explain:
        1. Why the risk was adjusted based on population context
        2. What the adjustment factors mean medically
        3. How population health patterns affect individual risk
        4. Clinical significance of the population-adjusted risk level
        5. Actionable recommendations based on this context

        Provide clear, medically sound explanation suitable for patient education.
        """
        
        try:
            explanation = self._generate_explanation(
                self.system_prompts["risk_analyzer"], 
                user_prompt
            )
            
            return {
                "explanation": explanation,
                "risk_level": risk_calibration.get('risk_level', 'Unknown'),
                "key_adjustments": risk_calibration.get('adjustment_factors', {}),
                "population_context": population_context.get('country', 'Unknown'),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "error": f"Failed to explain risk calibration: {str(e)}",
                "simple_explanation": f"Your risk has been adjusted to {risk_calibration.get('risk_level', 'Unknown').lower()} based on population health patterns in {population_context.get('country', 'your region')}.",
                "timestamp": time.time()
            }
    
    def generate_personalized_recommendations(self,
                                            prediction_result: Dict,
                                            user_data: Dict,
                                            existing_recommendations: List[Dict],
                                            population_context: Optional[Dict] = None) -> Dict:
        """
        Generate personalized, medically-grounded recommendations
        
        Args:
            prediction_result: ML prediction results
            user_data: User health data
            existing_recommendations: Algorithm-generated recommendations
            population_context: Population health context
            
        Returns:
            Enhanced medical recommendations
        """
        
        recommendations_text = "\n".join([
            f"- {rec['category']}: {rec['recommendation']} (Priority: {rec['priority']})"
            for rec in existing_recommendations
        ])
        
        user_prompt = f"""
        Please enhance and provide medical context for these sleep health recommendations:

        PATIENT PROFILE:
        {json.dumps(user_data, indent=2)}

        PREDICTION:
        - Disorder Risk: {prediction_result.get('predicted_class', 'Unknown')}
        - Confidence: {prediction_result.get('confidence', 0)*100:.1f}%

        CURRENT RECOMMENDATIONS:
        {recommendations_text}

        {self._build_medical_context(prediction_result, user_data, population_context)}

        Please provide:
        1. ENHANCED RECOMMENDATIONS: Improve existing recommendations with medical evidence
        2. PRIORITIZATION: Medical reasoning for recommendation priorities
        3. IMPLEMENTATION GUIDANCE: How to safely implement recommendations
        4. TIMELINE: Suggested timeline for implementing changes
        5. MONITORING: What signs to watch for during implementation
        6. RED FLAGS: Warning signs that require immediate medical attention

        Format as actionable, medically-sound guidance with clear priorities.
        """
        
        try:
            enhanced_recommendations = self._generate_explanation(
                self.system_prompts["health_educator"],
                user_prompt
            )
            
            return {
                "enhanced_recommendations": enhanced_recommendations,
                "original_count": len(existing_recommendations),
                "medical_priority_order": self._extract_priorities(enhanced_recommendations),
                "implementation_timeline": self._extract_timeline(enhanced_recommendations),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "error": f"Failed to enhance recommendations: {str(e)}",
                "fallback": "Please consult with a healthcare provider to discuss your sleep health recommendations and develop a personalized treatment plan.",
                "timestamp": time.time()
            }
    
    def _generate_explanation(self, system_prompt: str, user_prompt: str) -> str:
        """Generate explanation using Groq API"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_completion_tokens"],
                top_p=0.9,
                stream=False
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    
    def _build_medical_context(self, 
                             prediction_result: Dict,
                             user_data: Dict,
                             population_context: Optional[Dict]) -> str:
        """Build comprehensive medical context for prompts"""
        
        context_parts = []
        
        if population_context:
            context_parts.append(f"""
        POPULATION CONTEXT ({population_context.get('country', 'Unknown')}):
        - Life Expectancy: {population_context.get('health_indicators', {}).get('Life expectancy ', 'Unknown')}
        - Population BMI: {population_context.get('health_indicators', {}).get(' BMI ', 'Unknown')}
        - Development Status: {population_context.get('status', 'Unknown')}
        - Adult Mortality Rate: {population_context.get('health_indicators', {}).get('Adult Mortality', 'Unknown')}
            """)
        
        # Add relevant medical context based on prediction
        predicted_disorder = prediction_result.get('predicted_class', '')
        if predicted_disorder == 'Sleep Apnea':
            context_parts.append("""
        MEDICAL CONTEXT - SLEEP APNEA:
        Sleep apnea is a serious sleep disorder characterized by repeated breathing interruptions during sleep.
        Risk factors include obesity, age, male gender, and anatomical factors.
        Untreated sleep apnea increases cardiovascular disease risk, diabetes, and cognitive impairment.
            """)
        elif predicted_disorder == 'Insomnia':
            context_parts.append("""
        MEDICAL CONTEXT - INSOMNIA:
        Insomnia involves difficulty falling asleep, staying asleep, or early morning awakening.
        Contributing factors include stress, anxiety, poor sleep hygiene, and medical conditions.
        Chronic insomnia affects immune function, mental health, and overall quality of life.
            """)
        
        return "\n".join(context_parts)
    
    def _structure_explanation(self, 
                             explanation: str,
                             prediction_result: Dict,
                             user_data: Dict) -> Dict:
        """Structure the generated explanation into organized components"""
        
        return {
            "full_explanation": explanation,
            "prediction_summary": {
                "disorder": prediction_result.get('predicted_class', 'Unknown'),
                "confidence": f"{prediction_result.get('confidence', 0)*100:.1f}%",
                "risk_level": self._determine_risk_level(prediction_result.get('confidence', 0))
            },
            "key_risk_factors": self._extract_risk_factors(user_data),
            "explanation_type": "comprehensive",
            "medical_disclaimer": "This explanation is for educational purposes only and does not constitute medical advice. Please consult with a qualified healthcare provider for proper medical evaluation and treatment recommendations.",
            "timestamp": time.time(),
            "generated_by": f"Groq {self.model}"
        }
    
    def _generate_fallback_explanation(self, prediction_result: Dict, user_data: Dict) -> str:
        """Generate simple fallback explanation when API fails"""
        
        disorder = prediction_result.get('predicted_class', 'Unknown')
        confidence = prediction_result.get('confidence', 0) * 100
        
        fallback = f"""
        Based on your health profile, our analysis indicates a {confidence:.1f}% likelihood of {disorder}.
        
        Key factors contributing to this assessment:
        - Sleep duration: {user_data.get('Sleep Duration', 'Unknown')} hours
        - BMI category: {user_data.get('BMI Category', 'Unknown')}
        - Activity level: {user_data.get('Physical Activity Level', 'Unknown')}
        - Stress level: {user_data.get('Stress Level', 'Unknown')}/10
        
        General recommendations:
        - Maintain consistent sleep schedule (7-9 hours nightly)
        - Regular physical activity and stress management
        - Consult healthcare provider for comprehensive evaluation
        
        Important: This is not medical advice. Please seek professional medical consultation.
        """
        
        return fallback.strip()
    
    def _determine_risk_level(self, confidence: float) -> str:
        """Determine risk level based on confidence score"""
        if confidence > 0.8:
            return "High"
        elif confidence > 0.6:
            return "Moderate"
        elif confidence > 0.4:
            return "Low-Moderate"
        else:
            return "Low"
    
    def _extract_risk_factors(self, user_data: Dict) -> List[str]:
        """Extract key risk factors from user data"""
        
        risk_factors = []
        
        # BMI risk
        bmi_category = user_data.get('BMI Category', '')
        if bmi_category in ['Overweight', 'Obese']:
            risk_factors.append(f"BMI: {bmi_category}")
        
        # Sleep duration risk
        sleep_duration = user_data.get('Sleep Duration', 7)
        if isinstance(sleep_duration, (int, float)) and sleep_duration < 6.5:
            risk_factors.append("Short sleep duration")
        
        # High stress
        stress_level = user_data.get('Stress Level', 5)
        if isinstance(stress_level, (int, float)) and stress_level > 6:
            risk_factors.append("High stress level")
        
        # Low activity
        activity_level = user_data.get('Physical Activity Level', 50)
        if isinstance(activity_level, (int, float)) and activity_level < 30:
            risk_factors.append("Low physical activity")
        
        # Age factor
        age = user_data.get('Age', 35)
        if isinstance(age, (int, float)) and age > 50:
            risk_factors.append("Age over 50")
        
        return risk_factors
    
    def _extract_priorities(self, explanation: str) -> List[str]:
        """Extract priority order from explanation text"""
        # Simple implementation - could be enhanced with NLP
        priorities = ["High Priority", "Medium Priority", "Low Priority"]
        found_priorities = []
        
        for priority in priorities:
            if priority.lower() in explanation.lower():
                found_priorities.append(priority)
        
        return found_priorities
    
    def _extract_timeline(self, explanation: str) -> Dict[str, str]:
        """Extract implementation timeline from explanation"""
        # Simple implementation - could be enhanced
        timeline = {
            "immediate": "Start implementing high-priority recommendations immediately",
            "short_term": "Implement lifestyle changes within 2-4 weeks",
            "long_term": "Evaluate progress and adjust recommendations after 3 months"
        }
        
        return timeline
    
    def test_connection(self) -> Dict:
        """Test Groq API connection and model availability"""
        
        try:
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Test connection. Respond with 'Connected' only."}
                ],
                max_tokens=10,
                temperature=0
            )
            
            return {
                "status": "success",
                "model": self.model,
                "response": test_response.choices[0].message.content,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }


def main():
    """Test Groq medical explainer functionality"""
    
    print("🧠 Testing Groq Medical Explainer")
    print("=" * 50)
    
    # Initialize explainer
    explainer = GroqMedicalExplainer()
    
    # Test connection
    print("🔌 Testing API connection...")
    connection_test = explainer.test_connection()
    print(f"Status: {connection_test['status']}")
    if connection_test['status'] == 'success':
        print(f"Response: {connection_test['response']}")
    else:
        print(f"Error: {connection_test['error']}")
        return
    
    # Test explanation generation
    print(f"\n📝 Testing medical explanation generation...")
    
    # Sample prediction result
    sample_prediction = {
        'predicted_class': 'Sleep Apnea',
        'confidence': 0.85,
        'probabilities': {
            'None': 0.15,
            'Sleep Apnea': 0.85,
            'Insomnia': 0.0
        }
    }
    
    # Sample user data
    sample_user = {
        'Age': 45,
        'Gender': 'Male',
        'BMI Category': 'Obese',
        'Sleep Duration': 5.5,
        'Quality of Sleep': 4,
        'Physical Activity Level': 25,
        'Stress Level': 8,
        'Occupation': 'Manager'
    }
    
    # Sample population context
    sample_population = {
        'country': 'United States of America',
        'status': 'Developed',
        'health_indicators': {
            'Life expectancy ': 79.3,
            ' BMI ': 69.6,
            'Adult Mortality': 13
        }
    }
    
    try:
        explanation = explainer.explain_sleep_disorder_prediction(
            sample_prediction,
            sample_user,
            sample_population,
            "comprehensive"
        )
        
        print("✅ Medical explanation generated successfully!")
        print(f"Prediction: {explanation['prediction_summary']['disorder']}")
        print(f"Confidence: {explanation['prediction_summary']['confidence']}")
        print(f"Risk Level: {explanation['prediction_summary']['risk_level']}")
        print(f"Key Risk Factors: {', '.join(explanation['key_risk_factors'])}")
        
        print(f"\n📄 Sample explanation (first 200 chars):")
        print(f"{explanation['full_explanation'][:200]}...")
        
    except Exception as e:
        print(f"❌ Error generating explanation: {e}")
    
    print(f"\n✅ Groq Medical Explainer testing complete!")
    
    return explainer


if __name__ == "__main__":
    explainer = main()
