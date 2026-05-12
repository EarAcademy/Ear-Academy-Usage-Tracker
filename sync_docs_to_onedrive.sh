#!/bin/bash
# ============================================================
# Sync CLAUDE.md + README.md to OneDrive Claude Projects folder
# ============================================================
# Run this anytime CLAUDE.md or README.md changes, so the
# OneDrive copy stays current for stakeholders.
#
# HOW TO RUN (paste in Terminal):
#   bash ~/Desktop/ear-academy-analytics/sync_docs_to_onedrive.sh
# ============================================================

SRC="$HOME/Desktop/ear-academy-analytics"
DST="$HOME/Library/CloudStorage/OneDrive-TheEarAcademy/Claude Projects/Ear Academy Dashboards"

# Sanity checks
if [ ! -d "$SRC" ]; then
  echo "❌ Source folder not found: $SRC"
  exit 1
fi
if [ ! -f "$SRC/CLAUDE.md" ] || [ ! -f "$SRC/README.md" ]; then
  echo "❌ CLAUDE.md or README.md missing from $SRC"
  exit 1
fi

mkdir -p "$DST" || { echo "❌ Could not create $DST"; exit 1; }

echo "Copying documentation to OneDrive..."
ditto "$SRC/CLAUDE.md"  "$DST/CLAUDE.md"  && echo "  ✅ CLAUDE.md"
ditto "$SRC/README.md"  "$DST/README.md"  && echo "  ✅ README.md"

echo ""
echo "Destination:"
echo "  $DST"
echo ""
echo "Files now in OneDrive:"
ls -lh "$DST" | tail -n +2 | awk '{printf "  %s  %s  %s\n", $5, $6" "$7" "$8, $9}'
echo ""
echo "OneDrive will sync these to the cloud automatically (usually within seconds)."
