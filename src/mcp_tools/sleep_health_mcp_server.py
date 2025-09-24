"""
Sleep Health & Life Expectancy Risk Coach MCP Server
Exposes comprehensive sleep health assessment tools via Model Context Protocol
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../api'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))

try:
    from complete_sleep_health_system import CompleteSleepHealthSystem
    from groq_explainer import GroqMedicalExplainer
    from who_integration import WHOHealthContext
except ImportError as e:
    print(f"Warning: Could not import sleep health modules: {e}")
    CompleteSleepHealthSystem = None
    GroqMedicalExplainer = None
    WHOHealthContext = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mcp_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sleep_health_mcp")

class SleepHealthMCPServer:
    """
    MCP Server for Sleep Health & Life Expectancy Risk Assessment
    
    Provides AI assistants with access to:
    - Sleep disorder prediction and quality assessment
    - WHO population health context
    - Medical explanations and risk analysis  
    - Prediction monitoring and logging
    """
    
    def __init__(self):
        """Initialize MCP server with sleep health system components"""
        
        self.server = Server("sleep-health-coach")
        self.system = None
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize sleep health system
        self._initialize_health_system()
        
        # Register tools
        self._register_tools()
        
        logger.info("Sleep Health MCP Server initialized")
    
    def _initialize_health_system(self):
        """Initialize the complete sleep health assessment system"""
        
        try:
            if CompleteSleepHealthSystem:
                self.system = CompleteSleepHealthSystem()
                logger.info("Complete sleep health system initialized")
            else:
                logger.warning("Sleep health system modules not available")
                self.system = None
        except Exception as e:
            logger.error(f"Failed to initialize sleep health system: {e}")
            self.system = None
    
    def _register_tools(self):
        """Register all MCP tools with the server"""
        
        # Tool 1: Sleep Prediction
        @self.server.call_tool()
        async def sleep_predict(arguments: dict) -> list[types.TextContent]:
            """
            Predict sleep disorder risk and sleep quality for an individual
            
            Args:
                user_data: Dictionary with user health and lifestyle data
                country: Optional country for population context (default: USA)
                include_explanation: Whether to include medical explanation (default: true)
                
            Returns:
                Comprehensive sleep health assessment with predictions and recommendations
            """
            return await self._handle_sleep_predict(arguments)
        
        # Tool 2: WHO Context
        @self.server.call_tool()
        async def context_who_indicators(arguments: dict) -> list[types.TextContent]:
            """
            Get WHO population health indicators and context for a country
            
            Args:
                country: Country name (must match WHO dataset naming)
                include_trends: Whether to include health trends analysis (default: true)
                years: Number of years for trend analysis (default: 10)
                
            Returns:
                Country health profile with indicators, benchmarks, and trends
            """
            return await self._handle_who_context(arguments)
        
        # Tool 3: Risk Explanation
        @self.server.call_tool()
        async def explain_risk_factors(arguments: dict) -> list[types.TextContent]:
            """
            Generate medical explanation for sleep health risk factors
            
            Args:
                prediction_result: Sleep disorder prediction results
                user_data: Individual health data
                population_context: Optional WHO population context
                explanation_type: Type of explanation (comprehensive, risk_focused, educational)
                
            Returns:
                Medical explanation with clinical reasoning and recommendations
            """
            return await self._handle_risk_explanation(arguments)
        
        # Tool 4: Prediction Monitoring
        @self.server.call_tool()
        async def monitor_log_prediction(arguments: dict) -> list[types.TextContent]:
            """
            Log prediction for monitoring and drift detection
            
            Args:
                assessment_result: Complete assessment result to log
                user_id: Optional user identifier
                session_id: Optional session identifier
                
            Returns:
                Logging confirmation with monitoring insights
            """
            return await self._handle_prediction_logging(arguments)
        
        # Tool 5: Country Comparison
        @self.server.call_tool()
        async def compare_countries(arguments: dict) -> list[types.TextContent]:
            """
            Compare sleep health risk across multiple countries
            
            Args:
                user_data: Individual health data
                countries: List of country names to compare
                
            Returns:
                Cross-country risk comparison and analysis
            """
            return await self._handle_country_comparison(arguments)
        
        # Tool 6: System Status
        @self.server.call_tool()
        async def system_status(arguments: dict) -> list[types.TextContent]:
            """
            Get system status and component availability
            
            Returns:
                Status of ML models, WHO data, Groq API, and system performance
            """
            return await self._handle_system_status(arguments)
        
        logger.info("All MCP tools registered successfully")
    
    async def _handle_sleep_predict(self, arguments: dict) -> list[types.TextContent]:
        """Handle sleep disorder prediction request"""
        
        try:
            # Extract arguments
            user_data = arguments.get("user_data", {})
            country = arguments.get("country", "United States of America")
            include_explanation = arguments.get("include_explanation", True)
            
            # Validate user data
            if not user_data:
                return [types.TextContent(
                    type="text",
                    text="Error: user_data is required for sleep prediction"
                )]
            
            # Log request
            logger.info(f"Sleep prediction request for country: {country}")
            
            if not self.system:
                return [types.TextContent(
                    type="text",
                    text="Error: Sleep health system not available"
                )]
            
            # Perform comprehensive assessment
            start_time = time.time()
            assessment = self.system.comprehensive_assessment(
                user_data=user_data,
                country=country,
                include_explanation=include_explanation
            )
            processing_time = time.time() - start_time
            
            # Format response
            response = {
                "tool": "sleep.predict",
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": round(processing_time * 1000, 2),
                "assessment": assessment
            }
            
            # Log successful prediction
            self._log_prediction(assessment, "sleep.predict")
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response, indent=2, default=str)
            )]
            
        except Exception as e:
            logger.error(f"Error in sleep prediction: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    async def _handle_who_context(self, arguments: dict) -> list[types.TextContent]:
        """Handle WHO health context request"""
        
        try:
            # Extract arguments
            country = arguments.get("country", "United States of America")
            include_trends = arguments.get("include_trends", True)
            years = arguments.get("years", 10)
            
            logger.info(f"WHO context request for: {country}")
            
            if not self.system or not self.system.who_context:
                return [types.TextContent(
                    type="text",
                    text="Error: WHO health context not available"
                )]
            
            # Get country context
            context = self.system.who_context.get_country_context(country)
            if not context:
                return [types.TextContent(
                    type="text",
                    text=f"Error: Country '{country}' not found in WHO dataset"
                )]
            
            # Get health trends if requested
            trends = None
            if include_trends:
                trends = self.system.who_context.get_health_trends(country, years)
            
            # Get available countries for reference
            available_countries = self.system.who_context.get_major_countries()
            
            response = {
                "tool": "context.who_indicators",
                "timestamp": datetime.now().isoformat(),
                "country": country,
                "context": context,
                "health_trends": trends,
                "available_major_countries": available_countries[:10]  # Top 10
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response, indent=2, default=str)
            )]
            
        except Exception as e:
            logger.error(f"Error in WHO context: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    async def _handle_risk_explanation(self, arguments: dict) -> list[types.TextContent]:
        """Handle medical risk explanation request"""
        
        try:
            # Extract arguments
            prediction_result = arguments.get("prediction_result", {})
            user_data = arguments.get("user_data", {})
            population_context = arguments.get("population_context")
            explanation_type = arguments.get("explanation_type", "comprehensive")
            
            logger.info(f"Risk explanation request: {explanation_type}")
            
            if not prediction_result:
                return [types.TextContent(
                    type="text",
                    text="Error: prediction_result is required for explanation"
                )]
            
            if not self.system or not self.system.groq_available:
                return [types.TextContent(
                    type="text",
                    text="Error: Groq medical explanation service not available"
                )]
            
            # Generate medical explanation
            explanation = self.system.groq_explainer.explain_sleep_disorder_prediction(
                prediction_result=prediction_result,
                user_data=user_data,
                population_context=population_context,
                explanation_type=explanation_type
            )
            
            response = {
                "tool": "explain.risk_factors",
                "timestamp": datetime.now().isoformat(),
                "explanation_type": explanation_type,
                "explanation": explanation
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response, indent=2, default=str)
            )]
            
        except Exception as e:
            logger.error(f"Error in risk explanation: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    async def _handle_prediction_logging(self, arguments: dict) -> list[types.TextContent]:
        """Handle prediction logging for monitoring"""
        
        try:
            # Extract arguments
            assessment_result = arguments.get("assessment_result", {})
            user_id = arguments.get("user_id", "anonymous")
            session_id = arguments.get("session_id", f"session_{int(time.time())}")
            
            if not assessment_result:
                return [types.TextContent(
                    type="text",
                    text="Error: assessment_result is required for logging"
                )]
            
            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "session_id": session_id,
                "assessment_id": assessment_result.get("assessment_id"),
                "prediction": {
                    "sleep_disorder": assessment_result.get("ml_predictions", {}).get("sleep_disorder"),
                    "sleep_quality": assessment_result.get("ml_predictions", {}).get("sleep_quality")
                },
                "risk_calibration": assessment_result.get("population_context", {}).get("risk_calibration"),
                "processing_time_ms": assessment_result.get("system_performance", {}).get("processing_time_ms"),
                "components_used": assessment_result.get("system_performance", {}).get("components_used")
            }
            
            # Write to log file
            log_file = self.logs_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Generate monitoring insights
            insights = await self._generate_monitoring_insights()
            
            response = {
                "tool": "monitor.log_prediction",
                "timestamp": datetime.now().isoformat(),
                "logged": True,
                "log_file": str(log_file),
                "session_id": session_id,
                "monitoring_insights": insights
            }
            
            logger.info(f"Prediction logged for session: {session_id}")
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response, indent=2, default=str)
            )]
            
        except Exception as e:
            logger.error(f"Error in prediction logging: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    async def _handle_country_comparison(self, arguments: dict) -> list[types.TextContent]:
        """Handle multi-country risk comparison"""
        
        try:
            # Extract arguments
            user_data = arguments.get("user_data", {})
            countries = arguments.get("countries", [])
            
            if not user_data:
                return [types.TextContent(
                    type="text",
                    text="Error: user_data is required for country comparison"
                )]
            
            if not countries:
                # Use default major countries
                countries = ["United States of America", "Germany", "Japan", "Australia", "Canada"]
            
            logger.info(f"Country comparison request for {len(countries)} countries")
            
            if not self.system:
                return [types.TextContent(
                    type="text",
                    text="Error: Sleep health system not available"
                )]
            
            # Perform country comparison
            comparison = self.system.compare_across_countries(user_data, countries)
            
            response = {
                "tool": "compare.countries",
                "timestamp": datetime.now().isoformat(),
                "comparison": comparison
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response, indent=2, default=str)
            )]
            
        except Exception as e:
            logger.error(f"Error in country comparison: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    async def _handle_system_status(self, arguments: dict) -> list[types.TextContent]:
        """Handle system status request"""
        
        try:
            # Check system components
            status = {
                "tool": "system.status",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "sleep_health_system": self.system is not None,
                    "ml_models": self.system.models_available if self.system else False,
                    "who_integration": self.system.who_context is not None if self.system else False,
                    "groq_api": self.system.groq_available if self.system else False
                },
                "performance": {
                    "server_uptime": time.time() - getattr(self, 'start_time', time.time()),
                    "logs_directory": str(self.logs_dir),
                    "log_files_count": len(list(self.logs_dir.glob("*.jsonl")))
                }
            }
            
            # Add component details if available
            if self.system:
                if hasattr(self.system, 'who_context'):
                    status["who_data"] = {
                        "countries_available": len(self.system.who_context.get_available_countries()),
                        "major_countries": len(self.system.who_context.get_major_countries())
                    }
                
                if hasattr(self.system, 'groq_explainer') and self.system.groq_available:
                    # Test Groq connection
                    groq_test = self.system.groq_explainer.test_connection()
                    status["groq_status"] = groq_test
            
            return [types.TextContent(
                type="text",
                text=json.dumps(status, indent=2, default=str)
            )]
            
        except Exception as e:
            logger.error(f"Error in system status: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    async def _generate_monitoring_insights(self) -> dict:
        """Generate monitoring insights from recent predictions"""
        
        try:
            # Read recent log files
            today = datetime.now().strftime('%Y%m%d')
            log_file = self.logs_dir / f"predictions_{today}.jsonl"
            
            if not log_file.exists():
                return {"message": "No predictions logged today"}
            
            # Count predictions and analyze patterns
            predictions_count = 0
            disorders_count = {"None": 0, "Sleep Apnea": 0, "Insomnia": 0}
            risk_levels = {"Low": 0, "Moderate": 0, "High": 0, "Very High": 0}
            
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        predictions_count += 1
                        
                        # Count disorder predictions
                        disorder = entry.get("prediction", {}).get("sleep_disorder", {}).get("predicted_class")
                        if disorder in disorders_count:
                            disorders_count[disorder] += 1
                        
                        # Count risk levels
                        risk_level = entry.get("risk_calibration", {}).get("risk_level")
                        if risk_level in risk_levels:
                            risk_levels[risk_level] += 1
                            
                    except json.JSONDecodeError:
                        continue
            
            return {
                "total_predictions_today": predictions_count,
                "disorder_distribution": disorders_count,
                "risk_level_distribution": risk_levels,
                "log_file": str(log_file)
            }
            
        except Exception as e:
            return {"error": f"Failed to generate insights: {str(e)}"}
    
    def _log_prediction(self, assessment: dict, tool_name: str):
        """Log prediction for monitoring"""
        
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "assessment_id": assessment.get("assessment_id"),
                "prediction": assessment.get("ml_predictions"),
                "risk_level": assessment.get("population_context", {}).get("risk_calibration", {}).get("risk_level"),
                "processing_time_ms": assessment.get("system_performance", {}).get("processing_time_ms")
            }
            
            # Write to daily log file
            log_file = self.logs_dir / f"tool_usage_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

    async def run(self):
        """Run the MCP server"""
        
        self.start_time = time.time()
        logger.info("Starting Sleep Health MCP Server")
        
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, 
                write_stream, 
                InitializationOptions(
                    server_name="sleep-health-coach",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )


async def main():
    """Main entry point"""
    server = SleepHealthMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
