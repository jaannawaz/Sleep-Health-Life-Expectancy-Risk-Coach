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
