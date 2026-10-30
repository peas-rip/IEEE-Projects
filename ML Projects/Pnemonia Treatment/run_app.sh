#!/usr/bin/env bash
# Pneumonia Detection — one-command launcher (Linux/macOS)
# Creates venv if missing, installs deps, starts the app, opens the browser.
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install -q -r requirements.txt
pip install -q -e ".[serve]"

# Build frontend if dist/ is missing
if [ ! -d "frontend/dist" ]; then
  if command -v npm &>/dev/null; then
    echo "Building frontend..."
    (cd frontend && npm install --silent && npm run build --silent)
  else
    echo "WARNING: frontend/dist missing and npm not found."
    echo "Install Node.js or pre-build the frontend before running this script."
    echo "The API is still available at http://localhost:8000/api/docs"
  fi
fi

echo ""
echo "Starting Pneumonia Detection on http://localhost:8000"
echo "Press Ctrl+C to stop."
echo ""

# Open browser after 2-second delay
(sleep 2 && (open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null || true)) &

python app.main
