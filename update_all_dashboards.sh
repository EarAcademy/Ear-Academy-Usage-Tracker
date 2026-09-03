#!/bin/bash
# ============================================================
# Ear Academy — Update ALL Dashboards
# ============================================================
# What this does:
#   1. Updates the Usage Dashboard  (index.html)
#   2. Updates the Sales Dashboard  (investor.html)
#   3. Updates the Velocity Dashboard (pipeline_velocity.html)
#   4. Updates the Follow-Up Tracker (followup_tracker.json — incremental,
#      only checks deals that newly dropped off the stale list today)
#   5. Commits + pushes everything to GitHub in ONE commit
#
# For a full weekly re-verification of the follow-up tracker (catches
# status drift on deals checked days ago), run separately:
#   python3 update_followup_tracker.py --full-reverify
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
followup_ok=0

echo ""
echo "=========================================================="
echo "  Ear Academy — Updating all dashboards + follow-up tracker"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

# ─── 0. Pull latest FIRST (guard against clobbering someone else's work) ──
# This repo now has more than one operator. If this machine's copy is behind
# and we rebuild + push without pulling, we'd silently overwrite whatever the
# other operator pushed (reverting their changes and any config like the
# removed schools). Pull first; if it fails (e.g. a genuine conflict), STOP
# rather than push a stale, conflicting state.
#
# EXCEPTION: index.html and the *.json data files are 100% machine-generated
# by the steps below — nothing in them is ever hand-authored. If the ONLY
# pull conflicts are in that known set of generated files, it's safe to just
# take the incoming (origin) version and move on: steps 1-4 below overwrite
# them with fresh correct content within seconds anyway. Conflicts in any
# OTHER file still stop the script for a human to look at, same as before.
GENERATED_FILES="index.html pipeline_data.json paying_schools.json velocity_data.json renewals_data.json followup_tracker.json"

echo ""
echo "[0/4] ⬇️  Pulling latest from GitHub before rebuilding..."
echo "----------------------------------------------------------"
if ! git pull origin main; then
  conflicted=$(git diff --name-only --diff-filter=U)
  non_generated=$(comm -23 <(echo "$conflicted" | sort) <(echo "$GENERATED_FILES" | tr ' ' '\n' | sort))

  if [ -n "$conflicted" ] && [ -z "$non_generated" ]; then
    echo ""
    echo "⚠️  Merge conflict, but only in auto-generated files — resolving automatically:"
    echo "$conflicted" | sed 's/^/    /'
    for f in $conflicted; do
      git checkout --theirs -- "$f"
      git add "$f"
    done
    if git commit -m "Merge: auto-resolved generated-file conflict (took incoming, will regenerate)"; then
      echo "✅ Auto-resolved — continuing with fresh regeneration below."
    else
      echo "❌ Could not complete the auto-merge commit — stopping. See error above."
      exit 1
    fi
  else
    echo ""
    echo "❌ git pull failed — NOT rebuilding or pushing."
    if [ -n "$non_generated" ]; then
      echo "   Conflict includes non-generated file(s) that need a human look:"
      echo "$non_generated" | sed 's/^/     - /'
    fi
    echo "   Resolve the conflict/error above, then re-run. Nothing was published,"
    echo "   so the live dashboards are untouched."
    exit 1
  fi
fi

# ─── 1. Sales Dashboard (RUNS FIRST — writes paying_schools.json) ──
# Usage dashboard now reads paying_schools.json as the canonical roster of
# paying schools, so sales must run first so that JSON is fresh.
echo ""
echo "[1/4] 💰 Sales Dashboard (pulls live data from ActiveCampaign)"
echo "----------------------------------------------------------"
if python3 update_sales_dashboard.py; then
  echo "✅ Sales dashboard updated"
  sales_ok=1
else
  echo "❌ Sales dashboard FAILED — see error above"
fi

# ─── 2. Usage Dashboard ─────────────────────────────────────
echo ""
echo "[2/4] 📊 Usage Dashboard (reads daily_snapshots/ + paying_schools.json)"
echo "----------------------------------------------------------"
if python3 update_dashboard.py; then
  echo "✅ Usage dashboard updated"
  usage_ok=1
else
  echo "❌ Usage dashboard FAILED — see error above"
fi

# ─── 3. Velocity Dashboard ──────────────────────────────────
echo ""
echo "[3/4] ⚡ Velocity Dashboard (pulls deal data from ActiveCampaign)"
echo "----------------------------------------------------------"
if python3 update_velocity.py; then
  echo "✅ Velocity dashboard updated"
  velocity_ok=1
else
  echo "❌ Velocity dashboard FAILED — see error above"
fi

# ─── 4. Follow-Up Tracker (incremental — only checks NEW stale drop-offs) ──
# Must run AFTER update_velocity.py — it reads velocity_data.json's fresh
# stale_deals list to compute what's newly dropped off since the last run.
echo ""
echo "[4/4] 📋 Follow-Up Tracker (incremental check of newly-cleared stale deals)"
echo "----------------------------------------------------------"
if [ $velocity_ok -eq 1 ]; then
  if python3 update_followup_tracker.py; then
    echo "✅ Follow-up tracker updated"
    followup_ok=1
  else
    echo "❌ Follow-up tracker FAILED — see error above"
  fi
else
  echo "⏭️  Skipped — velocity dashboard didn't update, so stale list isn't fresh."
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
[ $followup_ok -eq 1 ] && parts+=("followup")

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
[ $followup_ok -eq 1 ] && echo "  ✅ Follow-Up Tracker"  || echo "  ❌ Follow-Up Tracker"
echo ""
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
echo ""
echo "Next: hard-refresh the dashboards in your browser (Cmd+Shift+R)"
echo "  Usage:     https://earacademy.github.io/Ear-Academy-Usage-Tracker/index.html"
echo "  Sales:     https://earacademy.github.io/Ear-Academy-Usage-Tracker/investor.html"
echo "  Velocity:  https://earacademy.github.io/Ear-Academy-Usage-Tracker/pipeline_velocity.html"
echo "(GitHub Pages can take 1-2 minutes to rebuild after the push.)"
