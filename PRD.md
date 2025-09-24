PRD: Sleep Health & Life Expectancy Risk Coach (MCP-Integrated)

1. Overview

This project develops a real-time AI health application that predicts sleep disorder risk and sleep quality using the Sleep Health & Lifestyle dataset (synthetic, individual-level) and contextualizes predictions with WHO Life Expectancy data (real, population-level).

The system is exposed through the Model Context Protocol (MCP), enabling seamless integration with AI assistants (e.g., ChatGPT, Cursor, VS Code agents). The app leverages the Groq API to provide clinical-style explanations and lifestyle guidance.

⸻

2. Goals & Objectives
	•	Provide personalized sleep disorder risk prediction in real time.
	•	Enrich results with country-level WHO health indicators.
	•	Expose predictions, context, and explanations as MCP tools so any MCP-compatible AI can use them.
	•	Integrate Groq API for medically grounded reasoning and recommendations.
	•	Ensure system is scalable, composable, and future-proof.

⸻

3. Target Users
	•	Students & Researchers: Learning AI + health data integration.
	•	General Users: Daily lifestyle-based risk and health benchmarking.
	•	Clinicians (Exploratory): Using explanations and WHO context for discussion and education.
	•	Developers/Builders: Demonstrating how synthetic + real health data can be wrapped in MCP tools.

⸻

4. Use Cases
	1.	Daily Risk Check
	•	User inputs lifestyle and sleep data.
	•	System predicts risk and explains drivers via Groq API.
	2.	Country Benchmarking
	•	Individual risk is compared against WHO indicators for the chosen country.
	3.	Longitudinal Monitoring
	•	Daily logs create 7-, 14-, or 30-day trends.
	4.	MCP-enabled Assistant Integration
	•	Any MCP-compatible agent (e.g., Cursor AI) can call tools like sleep.predict or explain.risk_factors.

⸻

5. Datasets
	•	Synthetic Dataset: Sleep Health & Lifestyle (age, gender, sleep duration, activity, stress, BMI, habits, occupation).
	•	Real Dataset: WHO Life Expectancy (life expectancy, adult mortality, BMI, income, schooling, country, year).
	•	External Source: Groq API (clinical reasoning and lifestyle recommendations).

⸻

6. Functional Requirements
	•	Prediction Tools
	•	Predict sleep disorder risk (binary classification).
	•	Estimate sleep quality (regression).
	•	Contextual Tools
	•	Fetch WHO indicators for country/year.
	•	Benchmark individual risk vs. population averages.
	•	Explanation Tools (Groq API)
	•	Translate model outputs into medical-style reasoning.
	•	Provide actionable recommendations tailored for patients or clinicians.
	•	MCP Integration
	•	Expose all functions as MCP tools (sleep.predict, context.who_indicators, explain.risk_factors, etc.).
	•	Define standardized input/output schemas for tool interoperability.
	•	Monitoring & Logging
	•	Record predictions and explanations for drift and fairness evaluation.

⸻

7. Non-Functional Requirements
	•	Performance: Responses under 500 ms including Groq API call.
	•	Reliability: Fallback when Groq API or WHO data unavailable.
	•	Scalability: MCP tools usable across multiple AI agents without modification.
	•	Compliance: Educational demo only; not medical advice.

⸻

8. Tech Stack
	•	Data & Modeling: Sleep dataset + WHO dataset.
	•	External API: Groq for reasoning.
	•	Service Layer: MCP-compliant tool interface.
	•	UI Layer: Optional Streamlit dashboard for direct user testing.
	•	Deployment: Local (for testing) and cloud containers (for sharing).

⸻

9. Deliverables
	•	Harmonized datasets (synthetic + WHO).
	•	Sleep disorder and sleep quality prediction models.
	•	Groq-powered explanation workflows.
	•	MCP server exposing tools (predict, context, explain, monitor).
	•	Documentation: schemas, usage guide, disclaimers.
	•	Evaluation report: metrics, calibration, fairness, explainability samples.

⸻

10. Milestones & Timeline

Week	Task	Deliverable
1	Data cleaning & harmonization	Unified dataset
2	Baseline prediction models	Classifier & regressor
3	WHO integration & calibration	Context-aware model
4	Groq API integration	Explanations + recommendations
5	MCP tool wrapping	Predict + Context + Explain as MCP tools
6	Demo & reporting	Dashboard + documentation + report


⸻

11. Risks & Limitations
	•	Synthetic dataset may not reflect real patients.
	•	WHO dataset is population-level, not individual.
	•	Groq integration depends on API availability.
	•	Outputs are educational only, not diagnostic.

⸻

12. Future Extensions
	•	Connect to wearables (Fitbit, Apple Watch, Oura) for real user data.
	•	Expand MCP toolset to cover diet, exercise, chronic conditions.
	•	Fine-tune Groq prompts for sleep medicine specialization.
	•	Deploy as a health companion assistant accessible across AI platforms via MCP.

⸻

13. Server Startup Instructions

### Prerequisites
- Python 3.8+ installed
- Virtual environment activated
- All dependencies installed

### Quick Start Commands

**Option 1: Automated Startup (Recommended)**
```bash
# Navigate to project directory
cd /Users/jaan/Desktop/New-Life

# Run automated startup script
./start_local.sh
```

**Option 2: Manual Startup**
```bash
# Navigate to project directory  
cd /Users/jaan/Desktop/New-Life

# Activate virtual environment
source venv/bin/activate

# Set Python path
export PYTHONPATH=$(pwd)

# Start Flask API (Terminal 1)
python src/api/flask_api.py

# Start Streamlit Dashboard (Terminal 2 - New terminal)
streamlit run dashboard/streamlit_app.py

# Start MCP Server (Terminal 3 - Optional, for AI assistants)
python src/mcp_tools/sleep_health_mcp_server.py
```

**Option 3: Docker Deployment**
```bash
# Navigate to deployment directory
cd /Users/jaan/Desktop/New-Life/deployment

# Run deployment script
./deploy.sh
```

### Access URLs
- **API Documentation**: http://localhost:5000
- **Interactive Dashboard**: http://localhost:8501  
- **API Health Check**: http://localhost:5000/api/health
- **Demo Prediction**: http://localhost:5000/api/demo

### Troubleshooting
- If ports are in use: `lsof -ti:5000 | xargs kill -9`
- If dependencies missing: `pip install -r requirements.txt`
- If virtual env issues: Recreate with `python -m venv venv`
- Check logs in `logs/` directory for detailed error information

### Stop Services
```bash
# If using automated startup
./stop_local.sh

# Manual stop
lsof -ti:5000,8501 | xargs kill -9
```

