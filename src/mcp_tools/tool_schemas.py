"""
MCP Tool Schemas for Sleep Health & Life Expectancy Risk Coach
Defines the input/output schemas for all MCP tools
"""

from typing import Dict, List, Optional, Any

# Tool Schemas
TOOL_SCHEMAS = {
    "sleep.predict": {
        "name": "sleep.predict",
        "description": "Predict sleep disorder risk and sleep quality for an individual",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_data": {
                    "type": "object",
                    "description": "User health and lifestyle data",
                    "properties": {
                        "Gender": {"type": "string", "enum": ["Male", "Female"], "description": "User gender"},
                        "Age": {"type": "integer", "minimum": 18, "maximum": 100, "description": "Age in years"},
                        "Occupation": {"type": "string", "description": "User occupation"},
                        "Sleep Duration": {"type": "number", "minimum": 1, "maximum": 12, "description": "Average sleep duration in hours"},
                        "Quality of Sleep": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Sleep quality rating (1-10)"},
                        "Physical Activity Level": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Physical activity level (0-100)"},
                        "Stress Level": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Stress level (1-10)"},
                        "BMI Category": {"type": "string", "enum": ["Normal", "Overweight", "Obese"], "description": "BMI category"},
                        "Blood Pressure": {"type": "string", "pattern": "^\\d{2,3}/\\d{2,3}$", "description": "Blood pressure (systolic/diastolic)"},
                        "Heart Rate": {"type": "integer", "minimum": 40, "maximum": 200, "description": "Resting heart rate (bpm)"},
                        "Daily Steps": {"type": "integer", "minimum": 0, "maximum": 50000, "description": "Average daily steps"}
                    },
                    "required": ["Gender", "Age", "BMI Category", "Sleep Duration", "Stress Level"]
                },
                "country": {
                    "type": "string",
                    "description": "Country for population context (default: United States of America)",
                    "default": "United States of America"
                },
                "include_explanation": {
                    "type": "boolean",
                    "description": "Whether to include medical explanation (default: true)",
                    "default": True
                }
            },
            "required": ["user_data"]
        }
    },
    
    "context.who_indicators": {
        "name": "context.who_indicators",
        "description": "Get WHO population health indicators and context for a country",
        "inputSchema": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "Country name (must match WHO dataset naming)"
                },
                "include_trends": {
                    "type": "boolean",
                    "description": "Whether to include health trends analysis (default: true)",
                    "default": True
                },
                "years": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 20,
                    "description": "Number of years for trend analysis (default: 10)",
                    "default": 10
                }
            },
            "required": ["country"]
        }
    },
    
    "explain.risk_factors": {
        "name": "explain.risk_factors",
        "description": "Generate medical explanation for sleep health risk factors",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prediction_result": {
                    "type": "object",
                    "description": "Sleep disorder prediction results",
                    "properties": {
                        "predicted_class": {"type": "string", "description": "Predicted sleep disorder"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Prediction confidence"},
                        "probabilities": {"type": "object", "description": "Class probabilities"}
                    },
                    "required": ["predicted_class", "confidence"]
                },
                "user_data": {
                    "type": "object",
                    "description": "Individual health data"
                },
                "population_context": {
                    "type": "object",
                    "description": "Optional WHO population context"
                },
                "explanation_type": {
                    "type": "string",
                    "enum": ["comprehensive", "risk_focused", "educational"],
                    "description": "Type of explanation (default: comprehensive)",
                    "default": "comprehensive"
                }
            },
            "required": ["prediction_result"]
        }
    },
    
    "monitor.log_prediction": {
        "name": "monitor.log_prediction",
        "description": "Log prediction for monitoring and drift detection",
        "inputSchema": {
            "type": "object",
            "properties": {
                "assessment_result": {
                    "type": "object",
                    "description": "Complete assessment result to log"
                },
                "user_id": {
                    "type": "string",
                    "description": "Optional user identifier",
                    "default": "anonymous"
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session identifier"
                }
            },
            "required": ["assessment_result"]
        }
    },
    
    "compare.countries": {
        "name": "compare.countries",
        "description": "Compare sleep health risk across multiple countries",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_data": {
                    "type": "object",
                    "description": "Individual health data"
                },
                "countries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of country names to compare",
                    "maxItems": 10
                }
            },
            "required": ["user_data"]
        }
    },
    
    "system.status": {
        "name": "system.status", 
        "description": "Get system status and component availability",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    }
}

# Example usage data for each tool
TOOL_EXAMPLES = {
    "sleep.predict": {
        "description": "Example sleep disorder prediction",
        "input": {
            "user_data": {
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
            },
            "country": "United States of America",
            "include_explanation": True
        }
    },
    
    "context.who_indicators": {
        "description": "Example WHO health context request",
        "input": {
            "country": "United States of America",
            "include_trends": True,
            "years": 10
        }
    },
    
    "explain.risk_factors": {
        "description": "Example medical explanation request",
        "input": {
            "prediction_result": {
                "predicted_class": "Sleep Apnea",
                "confidence": 0.85,
                "probabilities": {
                    "None": 0.15,
                    "Sleep Apnea": 0.85,
                    "Insomnia": 0.0
                }
            },
            "user_data": {
                "Age": 45,
                "BMI Category": "Obese",
                "Sleep Duration": 5.5,
                "Stress Level": 8
            },
            "explanation_type": "comprehensive"
        }
    },
    
    "monitor.log_prediction": {
        "description": "Example prediction logging",
        "input": {
            "assessment_result": {
                "assessment_id": "sleep_health_1234567890",
                "ml_predictions": {
                    "sleep_disorder": {
                        "predicted_class": "Sleep Apnea",
                        "confidence": 0.85
                    }
                }
            },
            "user_id": "user_123",
            "session_id": "session_456"
        }
    },
    
    "compare.countries": {
        "description": "Example country comparison",
        "input": {
            "user_data": {
                "Gender": "Female",
                "Age": 35,
                "BMI Category": "Normal",
                "Sleep Duration": 7.5,
                "Stress Level": 4
            },
            "countries": ["United States of America", "Germany", "Japan"]
        }
    },
    
    "system.status": {
        "description": "Example system status check",
        "input": {}
    }
}

# WHO country mappings for common variants
WHO_COUNTRY_MAPPINGS = {
    "USA": "United States of America",
    "US": "United States of America", 
    "United States": "United States of America",
    "UK": "United Kingdom",
    "Britain": "United Kingdom",
    "England": "United Kingdom",
    "Deutschland": "Germany",
    "Nederland": "Netherlands",
    "Holland": "Netherlands"
}

# Available major countries with complete WHO data
MAJOR_COUNTRIES = [
    "United States of America",
    "United Kingdom", 
    "Germany",
    "Japan",
    "Australia",
    "Canada",
    "France",
    "Italy",
    "Spain",
    "Netherlands",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Switzerland"
]

def validate_tool_input(tool_name: str, input_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate tool input against schema
    
    Args:
        tool_name: Name of the tool
        input_data: Input data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    if tool_name not in TOOL_SCHEMAS:
        return False, f"Unknown tool: {tool_name}"
    
    schema = TOOL_SCHEMAS[tool_name]["inputSchema"]
    
    # Basic validation (simplified - in production would use jsonschema)
    required_fields = schema.get("required", [])
    for field in required_fields:
        if field not in input_data:
            return False, f"Missing required field: {field}"
    
    # Validate user_data if present
    if "user_data" in input_data and tool_name == "sleep.predict":
        user_data = input_data["user_data"]
        user_schema = schema["properties"]["user_data"]
        
        for field in user_schema.get("required", []):
            if field not in user_data:
                return False, f"Missing required user_data field: {field}"
        
        # Validate specific fields
        if "Age" in user_data:
            age = user_data["Age"]
            if not isinstance(age, int) or age < 18 or age > 100:
                return False, "Age must be between 18 and 100"
        
        if "BMI Category" in user_data:
            bmi_cat = user_data["BMI Category"]
            if bmi_cat not in ["Normal", "Overweight", "Obese"]:
                return False, "BMI Category must be Normal, Overweight, or Obese"
        
        if "Sleep Duration" in user_data:
            duration = user_data["Sleep Duration"]
            if not isinstance(duration, (int, float)) or duration < 1 or duration > 12:
                return False, "Sleep Duration must be between 1 and 12 hours"
    
    return True, None

def normalize_country_name(country: str) -> str:
    """
    Normalize country name to WHO dataset format
    
    Args:
        country: Input country name
        
    Returns:
        Normalized country name
    """
    
    # Check direct mapping
    if country in WHO_COUNTRY_MAPPINGS:
        return WHO_COUNTRY_MAPPINGS[country]
    
    # Check if already in correct format
    if country in MAJOR_COUNTRIES:
        return country
    
    # Return as-is (will be validated by WHO integration)
    return country

def get_tool_documentation() -> Dict[str, Any]:
    """
    Get complete tool documentation
    
    Returns:
        Dictionary with tool schemas, examples, and usage information
    """
    
    return {
        "tools": TOOL_SCHEMAS,
        "examples": TOOL_EXAMPLES,
        "major_countries": MAJOR_COUNTRIES,
        "country_mappings": WHO_COUNTRY_MAPPINGS,
        "usage_notes": {
            "sleep.predict": "Primary tool for sleep health assessment. Requires minimal user data but works best with complete health profile.",
            "context.who_indicators": "Provides population health context. Country names must match WHO dataset exactly.",
            "explain.risk_factors": "Generates medical explanations. Requires prediction results from sleep.predict.",
            "monitor.log_prediction": "Logs predictions for monitoring. Use for tracking system performance and model drift.",
            "compare.countries": "Compares risk across countries. Limited to 10 countries per request for performance.",
            "system.status": "System health check. No parameters required."
        }
    }
