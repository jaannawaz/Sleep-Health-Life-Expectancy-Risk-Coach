"""
Flask REST API for Sleep Health & Life Expectancy Risk Coach
Provides REST endpoints for web applications and direct API access
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../mcp_tools'))

try:
    from complete_sleep_health_system import CompleteSleepHealthSystem
    from tool_schemas import validate_tool_input, normalize_country_name, MAJOR_COUNTRIES
    from config import API_CONFIG
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    CompleteSleepHealthSystem = None
    API_CONFIG = {"flask": {"host": "localhost", "port": 5000, "debug": True}}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sleep_health_api")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize sleep health system
sleep_system = None
try:
    if CompleteSleepHealthSystem:
        sleep_system = CompleteSleepHealthSystem()
        logger.info("Sleep health system initialized successfully")
    else:
        logger.warning("Sleep health system not available")
except Exception as e:
    logger.error(f"Failed to initialize sleep health system: {e}")

# API Documentation Template
API_DOCS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sleep Health API Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }
        .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
        .method { background: #28a745; color: white; padding: 4px 8px; border-radius: 3px; font-weight: bold; }
        .method.post { background: #ffc107; color: black; }
        code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; }
        .example { background: #e7f3ff; padding: 10px; border-radius: 5px; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Sleep Health & Life Expectancy Risk Coach API</h1>
        <p>Comprehensive sleep disorder prediction and health assessment API</p>
        <p><strong>Version:</strong> 1.0.0 | <strong>Status:</strong> {{ status }} | <strong>Components:</strong> {{ components }}</p>
    </div>
    
    <h2>🚀 Quick Start</h2>
    <p>Base URL: <code>http://{{ host }}:{{ port }}</code></p>
    <p>All endpoints return JSON responses with consistent error handling.</p>
    
    <h2>📊 Available Endpoints</h2>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> /api/health</h3>
        <p>System health check and component status</p>
        <div class="example">
            <strong>Response:</strong><br>
            <code>{"status": "healthy", "components": {...}, "timestamp": "..."}</code>
        </div>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /api/predict</h3>
        <p>Complete sleep health prediction with ML models, WHO context, and medical explanation</p>
        <div class="example">
            <strong>Request Body:</strong><br>
            <code>{"user_data": {...}, "country": "United States of America", "include_explanation": true}</code><br><br>
            <strong>Response:</strong><br>
            <code>{"assessment": {...}, "processing_time_ms": 850, "timestamp": "..."}</code>
        </div>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> /api/countries</h3>
        <p>Get list of available countries for WHO health context</p>
        <div class="example">
            <strong>Response:</strong><br>
            <code>{"countries": ["United States of America", ...], "count": 183}</code>
        </div>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /api/who-context</h3>
        <p>Get WHO population health indicators and trends for a country</p>
        <div class="example">
            <strong>Request Body:</strong><br>
            <code>{"country": "United States of America", "include_trends": true}</code>
        </div>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /api/explain</h3>
        <p>Generate medical explanation for sleep health predictions</p>
        <div class="example">
            <strong>Request Body:</strong><br>
            <code>{"prediction_result": {...}, "user_data": {...}, "explanation_type": "comprehensive"}</code>
        </div>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /api/compare-countries</h3>
        <p>Compare sleep health risk across multiple countries</p>
        <div class="example">
            <strong>Request Body:</strong><br>
            <code>{"user_data": {...}, "countries": ["USA", "Germany", "Japan"]}</code>
        </div>
    </div>
    
    <h2>🔧 Integration Examples</h2>
    <div class="example">
        <strong>JavaScript:</strong><br>
        <pre>
const response = await fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        user_data: {
            "Gender": "Male", "Age": 45, "BMI Category": "Overweight",
            "Sleep Duration": 6.0, "Stress Level": 7
        },
        country: "United States of America"
    })
});
const result = await response.json();
        </pre>
    </div>
    
    <h2>⚠️ Important Notes</h2>
    <ul>
        <li>All medical advice is for educational purposes only</li>
        <li>Predictions should not replace professional medical consultation</li>
        <li>WHO data covers 183 countries from 2000-2015</li>
        <li>Response times: Predictions &lt;500ms, Full assessment &lt;9s</li>
    </ul>
</body>
</html>
"""

@app.route('/')
def home():
    """API documentation homepage"""
    
    status = "healthy" if sleep_system else "limited"
    components = []
    
    if sleep_system:
        if hasattr(sleep_system, 'models_available') and sleep_system.models_available:
            components.append("ML Models")
        if hasattr(sleep_system, 'who_context') and sleep_system.who_context:
            components.append("WHO Integration")
        if hasattr(sleep_system, 'groq_available') and sleep_system.groq_available:
            components.append("Groq Medical AI")
    
    host = API_CONFIG.get("flask", {}).get("host", "localhost")
    port = API_CONFIG.get("flask", {}).get("port", 5000)
    
    return render_template_string(
        API_DOCS_TEMPLATE,
        status=status,
        components=", ".join(components) if components else "None",
        host=host,
        port=port
    )

@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    
    try:
        health_status = {
            "status": "healthy" if sleep_system else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "sleep_health_system": sleep_system is not None,
                "ml_models": sleep_system.models_available if sleep_system else False,
                "who_integration": sleep_system.who_context is not None if sleep_system else False,
                "groq_api": sleep_system.groq_available if sleep_system else False
            },
            "api_version": "1.0.0"
        }
        
        if sleep_system and hasattr(sleep_system, 'who_context'):
            health_status["data_coverage"] = {
                "countries_available": len(sleep_system.who_context.get_available_countries()),
                "major_countries": len(sleep_system.who_context.get_major_countries())
            }
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_sleep_health():
    """Complete sleep health prediction endpoint"""
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        # Extract parameters
        user_data = data.get('user_data', {})
        country = data.get('country', 'United States of America')
        include_explanation = data.get('include_explanation', True)
        
        # Validate user data
        if not user_data:
            return jsonify({"error": "user_data is required"}), 400
        
        # Validate input using MCP schemas
        is_valid, error_msg = validate_tool_input("sleep.predict", data)
        if not is_valid:
            return jsonify({"error": f"Invalid input: {error_msg}"}), 400
        
        # Normalize country name
        country = normalize_country_name(country)
        
        if not sleep_system:
            return jsonify({"error": "Sleep health system not available"}), 503
        
        # Perform assessment
        start_time = time.time()
        assessment = sleep_system.comprehensive_assessment(
            user_data=user_data,
            country=country,
            include_explanation=include_explanation
        )
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "assessment": assessment,
            "processing_time_ms": round(processing_time * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "api_version": "1.0.0"
        }
        
        logger.info(f"Prediction completed in {processing_time*1000:.1f}ms for {country}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/countries', methods=['GET'])
def get_countries():
    """Get available countries for WHO context"""
    
    try:
        if not sleep_system or not sleep_system.who_context:
            return jsonify({"error": "WHO integration not available"}), 503
        
        all_countries = sleep_system.who_context.get_available_countries()
        major_countries = sleep_system.who_context.get_major_countries()
        
        return jsonify({
            "countries": all_countries,
            "major_countries": major_countries,
            "count": len(all_countries),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Countries endpoint error: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/who-context', methods=['POST'])
def get_who_context():
    """Get WHO population health context for a country"""
    
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        country = data.get('country', 'United States of America')
        include_trends = data.get('include_trends', True)
        years = data.get('years', 10)
        
        # Normalize country name
        country = normalize_country_name(country)
        
        if not sleep_system or not sleep_system.who_context:
            return jsonify({"error": "WHO integration not available"}), 503
        
        # Get country context
        context = sleep_system.who_context.get_country_context(country)
        if not context:
            return jsonify({"error": f"Country '{country}' not found"}), 404
        
        # Get trends if requested
        trends = None
        if include_trends:
            trends = sleep_system.who_context.get_health_trends(country, years)
        
        response = {
            "country": country,
            "context": context,
            "health_trends": trends,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"WHO context error: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/explain', methods=['POST'])
def explain_risk_factors():
    """Generate medical explanation for sleep health predictions"""
    
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        prediction_result = data.get('prediction_result', {})
        user_data = data.get('user_data', {})
        population_context = data.get('population_context')
        explanation_type = data.get('explanation_type', 'comprehensive')
        
        if not prediction_result:
            return jsonify({"error": "prediction_result is required"}), 400
        
        if not sleep_system or not sleep_system.groq_available:
            return jsonify({"error": "Medical explanation service not available"}), 503
        
        # Generate explanation
        explanation = sleep_system.groq_explainer.explain_sleep_disorder_prediction(
            prediction_result=prediction_result,
            user_data=user_data,
            population_context=population_context,
            explanation_type=explanation_type
        )
        
        response = {
            "explanation": explanation,
            "explanation_type": explanation_type,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/compare-countries', methods=['POST'])
def compare_countries():
    """Compare sleep health risk across multiple countries"""
    
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        user_data = data.get('user_data', {})
        countries = data.get('countries', [])
        
        if not user_data:
            return jsonify({"error": "user_data is required"}), 400
        
        if not countries:
            # Use default major countries
            countries = MAJOR_COUNTRIES[:5]
        
        # Limit to 10 countries for performance
        countries = countries[:10]
        
        # Normalize country names
        countries = [normalize_country_name(country) for country in countries]
        
        if not sleep_system:
            return jsonify({"error": "Sleep health system not available"}), 503
        
        # Perform comparison
        comparison = sleep_system.compare_across_countries(user_data, countries)
        
        response = {
            "comparison": comparison,
            "countries_analyzed": len(countries),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Country comparison error: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/demo', methods=['GET'])
def demo_prediction():
    """Demo endpoint with sample prediction"""
    
    try:
        # Sample user data
        demo_user = {
            "Gender": "Male",
            "Age": 45,
            "Occupation": "Software Engineer",
            "Sleep Duration": 6.0,
            "Quality of Sleep": 5,
            "Physical Activity Level": 30,
            "Stress Level": 7,
            "BMI Category": "Overweight",
            "Blood Pressure": "135/85",
            "Heart Rate": 78,
            "Daily Steps": 5000
        }
        
        if not sleep_system:
            return jsonify({
                "demo_user": demo_user,
                "note": "Sleep health system not available - showing sample data only",
                "timestamp": datetime.now().isoformat()
            })
        
        # Perform demo assessment
        assessment = sleep_system.comprehensive_assessment(
            user_data=demo_user,
            country="United States of America",
            include_explanation=True
        )
        
        response = {
            "demo_user": demo_user,
            "assessment": assessment,
            "note": "This is a demonstration with sample data",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "/", "/api/health", "/api/predict", "/api/countries",
            "/api/who-context", "/api/explain", "/api/compare-countries", "/api/demo"
        ],
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

def main():
    """Run the Flask application"""
    
    host = API_CONFIG.get("flask", {}).get("host", "localhost")
    port = API_CONFIG.get("flask", {}).get("port", 5000)
    debug = API_CONFIG.get("flask", {}).get("debug", True)
    
    print(f"🚀 Starting Sleep Health API Server")
    print(f"📍 URL: http://{host}:{port}")
    print(f"📚 Documentation: http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    
    if sleep_system:
        print(f"✅ Sleep health system operational")
        print(f"   • ML Models: {'✅' if sleep_system.models_available else '❌'}")
        print(f"   • WHO Integration: {'✅' if sleep_system.who_context else '❌'}")
        print(f"   • Groq Medical AI: {'✅' if sleep_system.groq_available else '❌'}")
    else:
        print(f"⚠️ Sleep health system not available")
    
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()
