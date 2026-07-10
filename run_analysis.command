#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Ear Academy newsletter analysis..."
echo ""
python3 newsletter_analysis.py
echo ""
echo "Done! You can close this window."
