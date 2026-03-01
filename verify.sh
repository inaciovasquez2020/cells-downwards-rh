#!/usr/bin/env bash
set -e

echo "=== PYTHON CHECK ==="
python3 -m py_compile $(git ls-files '*.py')

if [ -d "experiments" ]; then
  echo "=== RUNNING EXPERIMENTS (if safe) ==="
fi

echo "=== CERTIFICATE PHASE ==="
python3 tools/make_cert.py > cert.json
python3 tools/verify_cert.py cert.json

echo "=== DONE ==="
