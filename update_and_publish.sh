#!/bin/bash

echo "🔄 Updating dashboard..."
python3 update_dashboard.py

echo "📝 Committing changes..."
git add index.html update_dashboard.py
git commit -m "Update dashboard - $(date '+%Y-%m-%d %H:%M')"

echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ Done! Live site will update in 30 seconds."
echo "🌐 View at: https://earacademy.github.io/Ear-Academy-Usage-Tracker/"
