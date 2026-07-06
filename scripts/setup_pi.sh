#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project: $PROJECT_ROOT"
echo "Architecture: $(uname -m)"
echo "Python: $(python3 --version)"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "Running tests..."
python -m pytest -q

echo
echo "Raspberry Pi environment ready."
echo "Activate it with: source .venv/bin/activate"
