# 🚀 Sleep Health Risk Coach - Deployment Guide

## Overview

This guide covers deployment options for the Sleep Health & Life Expectancy Risk Coach application.

## 🏗️ Architecture

### Components
- **Flask REST API** - Backend service with comprehensive endpoints
- **Streamlit Dashboard** - Interactive web interface for users
- **MCP Server** - AI assistant integration via Model Context Protocol

### Deployment Options
1. **Local Development** - Single machine with virtual environment
2. **Docker Compose** - Multi-container local/staging deployment
3. **Production** - Cloud deployment recommendations

## 🖥️ Local Development

### Prerequisites
- Python 3.8+ (recommended: 3.11)
- 4GB+ RAM
- 2GB+ disk space

### Quick Start
```bash
# Clone/navigate to project
cd New-Life

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start all services
./start_local.sh

# Access services
# Dashboard: http://localhost:8501
# API: http://localhost:5000
```

### Manual Service Start
```bash
# Activate environment
source venv/bin/activate
export PYTHONPATH=$(pwd)

# Start Flask API
python src/api/flask_api.py &

# Start Streamlit Dashboard
streamlit run dashboard/streamlit_app.py &

# Start MCP Server (optional)
python src/mcp_tools/sleep_health_mcp_server.py &
```

### Stop Services
```bash
./stop_local.sh
```

## 🐳 Docker Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 4GB+ disk space

### Production Deployment
```bash
# Deploy all services
cd deployment
./deploy.sh

# Services will be available at:
# - Dashboard: http://localhost:8501
# - API: http://localhost:5000  
# - MCP: localhost:3000
```

### Manual Docker Commands
```bash
# Build images
docker-compose -f deployment/docker-compose.yml build

# Start services
docker-compose -f deployment/docker-compose.yml up -d

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Stop services
docker-compose -f deployment/docker-compose.yml down
```

### Service Health Checks
```bash
# API health
curl http://localhost:5000/api/health

# Dashboard health
curl http://localhost:8501/_stcore/health

# View service status
docker-compose -f deployment/docker-compose.yml ps
```

## ☁️ Cloud Deployment

### AWS ECS (Recommended)
```yaml
# Example ECS task definition
{
  "family": "sleep-health-coach",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-repo/sleep-health-api:latest",
      "portMappings": [{"containerPort": 5000}],
      "environment": [
        {"name": "FLASK_ENV", "value": "production"}
      ]
    },
    {
      "name": "dashboard", 
      "image": "your-repo/sleep-health-dashboard:latest",
      "portMappings": [{"containerPort": 8501}]
    }
  ]
}
```

### Google Cloud Run
```bash
# Build and push images
docker build -t gcr.io/PROJECT_ID/sleep-health-api .
docker push gcr.io/PROJECT_ID/sleep-health-api

# Deploy API
gcloud run deploy sleep-health-api \
  --image gcr.io/PROJECT_ID/sleep-health-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi

# Deploy Dashboard
docker build -f Dockerfile.streamlit -t gcr.io/PROJECT_ID/sleep-health-dashboard .
docker push gcr.io/PROJECT_ID/sleep-health-dashboard

gcloud run deploy sleep-health-dashboard \
  --image gcr.io/PROJECT_ID/sleep-health-dashboard \
  --platform managed \
  --region us-central1 \
  --memory 2Gi
```

### Azure Container Instances
```bash
# Create resource group
az group create --name sleep-health-rg --location eastus

# Deploy API container
az container create \
  --resource-group sleep-health-rg \
  --name sleep-health-api \
  --image your-registry/sleep-health-api:latest \
  --ports 5000 \
  --memory 2 \
  --cpu 1

# Deploy Dashboard container
az container create \
  --resource-group sleep-health-rg \
  --name sleep-health-dashboard \
  --image your-registry/sleep-health-dashboard:latest \
  --ports 8501 \
  --memory 2 \
  --cpu 1
```

## 🔧 Configuration

### Environment Variables
```bash
# Flask API
FLASK_ENV=production              # development/production
FLASK_DEBUG=False                 # True for development
PYTHONPATH=/app                   # Application root path

# Groq API
GROQ_API_KEY=your_api_key        # Required for medical explanations

# Streamlit
STREAMLIT_SERVER_PORT=8501        # Port for dashboard
STREAMLIT_SERVER_HEADLESS=true   # Run without browser
```

### Groq API Configuration
1. Obtain API key from Groq
2. Update `config/config.py`:
```python
API_CONFIG = {
    "groq": {
        "api_key": "your_groq_api_key_here",
        "model": "openai/gpt-oss-120b"
    }
}
```

### Data Files Setup
Required data files (included in repo):
- `data/processed/who_life_expectancy_key.csv` - WHO population data
- `src/models/*.joblib` - Trained ML models
- `src/models/performance_metrics.json` - Model performance data

## 🔍 Monitoring & Logging

### Application Logs
```bash
# Local development
tail -f logs/flask-api.log
tail -f logs/streamlit-dashboard.log

# Docker deployment
docker-compose -f deployment/docker-compose.yml logs -f api
docker-compose -f deployment/docker-compose.yml logs -f dashboard
```

### Health Monitoring
```bash
# API health endpoint
curl http://localhost:5000/api/health

# Expected response:
{
  "status": "healthy",
  "components": {
    "sleep_health_system": true,
    "ml_models": true,
    "who_integration": true,
    "groq_api": true
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Performance Monitoring
- **API Response Times**: <500ms for predictions
- **Memory Usage**: ~1GB typical, 2GB peak
- **CPU Usage**: <50% normal load
- **Disk Usage**: ~500MB for models and data

## 🤖 AI Assistant Integration

### Claude Desktop Configuration
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "sleep-health-coach": {
      "command": "python",
      "args": ["src/mcp_tools/sleep_health_mcp_server.py"],
      "cwd": "/path/to/New-Life"
    }
  }
}
```

### MCP Server Standalone
```bash
# Start MCP server
python src/mcp_tools/sleep_health_mcp_server.py

# Test MCP tools
python src/mcp_tools/test_mcp_client.py
```

## 🔒 Security Considerations

### API Security
- **CORS**: Configured for cross-origin requests
- **Input Validation**: Comprehensive schema validation
- **Error Handling**: No sensitive data in error messages
- **Rate Limiting**: Implement in production load balancer

### Data Privacy
- **No PII Storage**: User data processed in memory only
- **Anonymized Logging**: Personal identifiers removed
- **Medical Disclaimers**: All outputs include appropriate warnings

### Production Hardening
```bash
# Disable debug mode
export FLASK_ENV=production
export FLASK_DEBUG=False

# Use production WSGI server
pip install gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 src.api.flask_api:app
```

## 📊 Performance Tuning

### Resource Requirements
- **Minimum**: 2GB RAM, 1 CPU core
- **Recommended**: 4GB RAM, 2 CPU cores
- **High Load**: 8GB RAM, 4 CPU cores

### Optimization Settings
```python
# Flask API optimization
app.config['JSON_SORT_KEYS'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 86400

# Streamlit optimization
st.set_page_config(
    page_title="Sleep Health",
    layout="wide",
    initial_sidebar_state="collapsed"
)
```

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Fix Python path
export PYTHONPATH=$(pwd)

# Verify imports
python -c "from src.models.complete_sleep_health_system import CompleteSleepHealthSystem"
```

**2. Missing Data Files**
```bash
# Check required files
ls data/processed/who_life_expectancy_key.csv
ls src/models/*.joblib

# System will run in limited mode if files missing
```

**3. Groq API Issues**
```bash
# Test API connection
python -c "
from src.api.groq_explainer import GroqMedicalExplainer
explainer = GroqMedicalExplainer()
print(explainer.test_connection())
"
```

**4. Port Conflicts**
```bash
# Find processes using ports
lsof -i :5000
lsof -i :8501

# Kill processes
kill $(lsof -t -i:5000)
```

### Support
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Complete API docs at http://localhost:5000
- **Logs**: Check application logs for detailed error information

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Groq API key configured
- [ ] Data files present
- [ ] Model files available

### Local Development
- [ ] Virtual environment activated
- [ ] `./start_local.sh` runs successfully
- [ ] API accessible at http://localhost:5000
- [ ] Dashboard accessible at http://localhost:8501
- [ ] Health checks pass

### Production Deployment
- [ ] Docker/Docker Compose installed
- [ ] `./deploy.sh` runs successfully
- [ ] All containers healthy
- [ ] External access configured
- [ ] Monitoring set up
- [ ] Backup strategy in place

### AI Assistant Integration
- [ ] MCP server starts successfully
- [ ] MCP tools test passes
- [ ] Claude Desktop configured (if using)
- [ ] Custom integration tested

---

**Status**: ✅ Production Ready | All components validated | Full deployment support available
