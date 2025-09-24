#!/bin/bash

# Sleep Health & Life Expectancy Risk Coach
# Production Deployment Script

set -e

echo "🚀 Sleep Health Risk Coach - Production Deployment"
echo "=================================================="

# Check if required files exist
echo "📋 Checking deployment requirements..."

required_files=(
    "requirements.txt"
    "Dockerfile"
    "Dockerfile.streamlit" 
    "src/api/flask_api.py"
    "dashboard/streamlit_app.py"
    "src/mcp_tools/sleep_health_mcp_server.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done

echo "✅ All required files present"

# Check if data files exist
echo "📊 Checking data files..."

data_files=(
    "data/processed/who_life_expectancy_key.csv"
    "src/models/random_forest_model.joblib"
    "src/models/xgboost_model.joblib"
    "src/models/preprocessors.joblib"
)

missing_data_files=0
for file in "${data_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️ Data file missing: $file"
        missing_data_files=$((missing_data_files + 1))
    fi
done

if [ $missing_data_files -gt 0 ]; then
    echo "⚠️ Warning: $missing_data_files data files missing. System will run in limited mode."
else
    echo "✅ All data files present"
fi

# Set deployment environment
export FLASK_ENV=production
export PYTHONPATH=$(pwd)

# Create necessary directories
echo "📁 Creating deployment directories..."
mkdir -p logs
mkdir -p data/processed
mkdir -p deployment/backups
echo "✅ Directories created"

# Build and start services
echo "🐳 Building and starting Docker services..."

# Stop existing containers
docker-compose -f deployment/docker-compose.yml down

# Build images
docker-compose -f deployment/docker-compose.yml build

# Start services
docker-compose -f deployment/docker-compose.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Health checks
echo "🔍 Running health checks..."

# Check API
api_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/health || echo "000")
if [ "$api_health" = "200" ]; then
    echo "✅ API service healthy"
else
    echo "❌ API service not responding (HTTP $api_health)"
fi

# Check Dashboard
dashboard_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 || echo "000")
if [ "$dashboard_health" = "200" ]; then
    echo "✅ Dashboard service healthy"
else
    echo "❌ Dashboard service not responding (HTTP $dashboard_health)"
fi

# Display deployment status
echo ""
echo "🎉 DEPLOYMENT COMPLETE"
echo "======================"
echo "🌐 Services:"
echo "  • API Documentation: http://localhost:5000"
echo "  • Interactive Dashboard: http://localhost:8501"
echo "  • MCP Server: localhost:3000 (for AI assistants)"
echo ""
echo "📊 Service Status:"
docker-compose -f deployment/docker-compose.yml ps

echo ""
echo "📝 Next Steps:"
echo "  1. Test the API at http://localhost:5000"
echo "  2. Use the dashboard at http://localhost:8501"
echo "  3. Configure AI assistants with MCP server"
echo "  4. Monitor logs: docker-compose -f deployment/docker-compose.yml logs -f"
echo ""
echo "🛑 To stop services: docker-compose -f deployment/docker-compose.yml down"
