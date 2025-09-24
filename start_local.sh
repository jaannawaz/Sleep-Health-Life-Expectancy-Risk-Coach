#!/bin/bash

# Sleep Health & Life Expectancy Risk Coach
# Local Development Startup Script

set -e

echo "🏥 Sleep Health Risk Coach - Local Development"
echo "============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import flask, streamlit, groq, pandas, scikit_learn" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
}

echo "✅ Dependencies verified"

# Set environment variables
export PYTHONPATH=$(pwd)
export FLASK_ENV=development
export FLASK_DEBUG=True

# Create logs directory
mkdir -p logs

# Function to start service in background
start_service() {
    local service_name=$1
    local command=$2
    local port=$3
    local log_file="logs/${service_name}.log"
    
    echo "🚀 Starting $service_name on port $port..."
    
    # Kill existing process on port if any
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
    
    # Start service
    nohup $command > $log_file 2>&1 &
    local pid=$!
    echo $pid > "logs/${service_name}.pid"
    
    # Wait a moment and check if process is still running
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        echo "✅ $service_name started (PID: $pid)"
    else
        echo "❌ $service_name failed to start. Check $log_file"
        return 1
    fi
}

# Start services
echo ""
echo "🎯 Starting Sleep Health Services..."
echo "===================================="

# Start Flask API
start_service "flask-api" "python src/api/flask_api.py" 5000

# Start Streamlit Dashboard
start_service "streamlit-dashboard" "streamlit run dashboard/streamlit_app.py --server.port 8501 --server.headless true" 8501

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to initialize..."
sleep 5

# Health checks
echo ""
echo "🔍 Running health checks..."

# Check Flask API
api_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/health 2>/dev/null || echo "000")
if [ "$api_status" = "200" ]; then
    echo "✅ Flask API: Healthy (http://localhost:5000)"
else
    echo "❌ Flask API: Not responding"
fi

# Check Streamlit
streamlit_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 2>/dev/null || echo "000")
if [ "$streamlit_status" = "200" ]; then
    echo "✅ Streamlit Dashboard: Healthy (http://localhost:8501)"
else
    echo "❌ Streamlit Dashboard: Not responding"
fi

# Display service information
echo ""
echo "🎉 LOCAL DEVELOPMENT ENVIRONMENT READY"
echo "====================================="
echo ""
echo "🌐 Available Services:"
echo "  • 📚 API Documentation: http://localhost:5000"
echo "  • 🎛️ Interactive Dashboard: http://localhost:8501"
echo "  • 🔧 API Health Check: http://localhost:5000/api/health"
echo ""

echo "🧪 Quick Test Commands:"
echo '  • Test API: curl http://localhost:5000/api/health'
echo '  • Demo Prediction: curl http://localhost:5000/api/demo'
echo ""

echo "📋 Development Commands:"
echo "  • View API logs: tail -f logs/flask-api.log"
echo "  • View Dashboard logs: tail -f logs/streamlit-dashboard.log"
echo "  • Stop services: ./stop_local.sh"
echo ""

echo "🤖 MCP Integration:"
echo "  • Start MCP server: python src/mcp_tools/sleep_health_mcp_server.py"
echo "  • Test MCP tools: python src/mcp_tools/test_mcp_client.py"
echo ""

# Create stop script
cat > stop_local.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping Sleep Health Services..."

# Kill services by PID files
for service in flask-api streamlit-dashboard; do
    pid_file="logs/${service}.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            echo "✅ Stopped $service (PID: $pid)"
        fi
        rm -f "$pid_file"
    fi
done

# Kill by port as backup
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

echo "🏁 All services stopped"
EOF

chmod +x stop_local.sh

echo "✨ Development environment is ready!"
echo "Open http://localhost:8501 in your browser to start using the dashboard."
