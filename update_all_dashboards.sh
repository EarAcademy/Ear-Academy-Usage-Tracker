#!/bin/bash
# ============================================================
# Ear Academy — Update ALL Dashboards
# ============================================================
# What this does:
#   1. Updates the Usage Dashboard  (index.html)
#   2. Updates the Sales Dashboard  (investor.html)
#   3. Updates the Velocity Dashboard (pipeline_velocity.html)
#   4. Commits + pushes everything to GitHub in ONE commit
#
# Each Python step is independent — if one fails, the others
# still run, and you get a clear summary at the end.
#
# HOW TO RUN (paste in Terminal):
#   bash ~/Desktop/ear-academy-analytics/update_all_dashboards.sh
#
# AFTER IT FINISHES:
#   Hard-refresh the dashboards in your browser (Cmd+Shift+R):
#     - https://earacademy.github.io/Ear-Academy-Usage-Tracker/index.html
#     - https://earacademy.github.io/Ear-Academy-Usage-Tracker/investor.html
#     - https://earacademy.github.io/Ear-Academy-Usage-Tracker/pipeline_velocity.html
#   GitHub Pages can take 1-2 minutes to rebuild after the push.
# ============================================================

cd ~/Desktop/ear-academy-analytics || {
  echo "❌ Cannot find ~/Desktop/ear-academy-analytics — script aborted."
  exit 1
}

# Track which updates succeeded
usage_ok=0
sales_ok=0
velocity_ok=0

echo ""
echo "=========================================================="
echo "  Ear Academy — Updating all 3 dashboards"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

# ─── 1. Sales Dashboard (RUNS FIRST — writes paying_schools.json) ──
# Usage dashboard now reads paying_schools.json as the canonical roster of
# paying schools, so sales must run first so that JSON is fresh.
echo ""
echo "[1/3] 💰 Sales Dashboard (pulls live data from ActiveCampaign)"
echo "----------------------------------------------------------"
if python3 update_sales_dashboard.py; then
  echo "✅ Sales dashboard updated"
  sales_ok=1
else
  echo "❌ Sales dashboard FAILED — see error above"
fi

# ─── 2. Usage Dashboard ─────────────────────────────────────
echo ""
echo "[2/3] 📊 Usage Dashboard (reads daily_snapshots/ + paying_schools.json)"
echo "----------------------------------------------------------"
if python3 update_dashboard.py; then
  echo "✅ Usage dashboard updated"
  usage_ok=1
else
  echo "❌ Usage dashboard FAILED — see error above"
fi

# ─── 3. Velocity Dashboard ──────────────────────────────────
echo ""
echo "[3/3] ⚡ Velocity Dashboard (pulls deal data from ActiveCampaign)"
echo "----------------------------------------------------------"
if python3 update_velocity.py; then
  echo "✅ Velocity dashboard updated"
  velocity_ok=1
else
  echo "❌ Velocity dashboard FAILED — see error above"
fi

# ─── Git commit + push ──────────────────────────────────────
echo ""
echo "=========================================================="
echo "  Pushing changes to GitHub"
echo "=========================================================="

# Build a commit message that reflects what actually ran
parts=()
[ $usage_ok    -eq 1 ] && parts+=("usage")
[ $sales_ok    -eq 1 ] && parts+=("sales")
[ $velocity_ok -eq 1 ] && parts+=("velocity")

if [ ${#parts[@]} -eq 0 ]; then
  echo "⚠️  All three updates failed — nothing to commit. Check the errors above."
else
  IFS='+' eval 'label="${parts[*]}"'
  msg="Update dashboards [$label] $(date '+%Y-%m-%d %H:%M')"

  git add -A
  if git diff --cached --quiet; then
    echo "ℹ️  No changes to commit (dashboards already up to date)."
  else
    git commit -m "$msg" && git push origin main
  fi
fi

# ─── Summary ────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "  SUMMARY"
echo "=========================================================="
[ $usage_ok    -eq 1 ] && echo "  ✅ Usage Dashboard"    || echo "  ❌ Usage Dashboard"
[ $sales_ok    -eq 1 ] && echo "  ✅ Sales Dashboard"    || echo "  ❌ Sales Dashboard"
[ $velocity_ok -eq 1 ] && echo "  ✅ Velocity Dashboard" || echo "  ❌ Velocity Dashboard"
echo ""
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
echo ""
echo "Next: hard-refresh the dashboards in your browser (Cmd+Shift+R)"
echo "  Usage:     https://earacademy.github.io/Ear-Academy-Usage-Tracker/index.html"
echo "  Sales:     https://earacademy.github.io/Ear-Academy-Usage-Tracker/investor.html"
echo "  Velocity:  https://earacademy.github.io/Ear-Academy-Usage-Tracker/pipeline_velocity.html"
echo "(GitHub Pages can take 1-2 minutes to rebuild after the push.)"
