#!/usr/bin/env python3
"""
Ear Academy — Stale Pipeline Follow-Up Tracker
================================================
Tracks every deal that has EVER left the New Lead stale list (90d+) since
the cleanup began, and keeps its current AC status up to date — catching
things a simple list-diff can't, like a deal marked Lost and manually
reopened to "reset" it.

Two modes:
  Incremental (default): only checks deals that are NEW since the last run
    (dropped off the stale list today but weren't tracked yet). Cheap —
    typically 10-30 AC calls per day.
  --full-reverify: re-checks EVERY tracked deal's current status, to catch
    drift (e.g. a "Long Term Interest" deal that has since moved or gone
    cold again). Expensive — one AC call per tracked deal (100-200+).
    Run this weekly, not daily.

SAFE BY DESIGN:
  ✅  Writes ONLY to followup_tracker.json
  🚫  Never touches any HTML file

Must run AFTER update_velocity.py in the same session (reads its output).

Run as part of the daily update:
  python3 update_followup_tracker.py                    # incremental (daily)
  python3 update_followup_tracker.py --full-reverify     # deep sweep (weekly)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def strf(dt, fmt):
    if '%-d' in fmt:
        fmt = fmt.replace('%-d', str(dt.day))
    if '%-I' in fmt:
        fmt = fmt.replace('%-I', str((dt.hour - 1) % 12 + 1))
    return dt.strftime(fmt)


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import AC_API_KEY, AC_BASE_URL
except ImportError:
    print("❌  Could not find config.py. Make sure it is in the same folder as this script.")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("❌  'requests' library not found. Run: pip3 install requests --break-system-packages")
    sys.exit(1)

VELOCITY_FILE = SCRIPT_DIR / "velocity_data.json"
TRACKER_FILE = SCRIPT_DIR / "followup_tracker.json"
TODAY = datetime.now(timezone.utc)
TODAY_LABEL = strf(TODAY, "%-d %b %Y at %H:%M")

HEADERS = {
    "Api-Token": AC_API_KEY,
    "Content-Type": "application/json",
}


def get(endpoint, params=None):
    url = f"{AC_BASE_URL.rstrip('/')}/api/3/{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_deal_status(deal_id):
    """Fetch a single deal's current group/stage/status. Returns not_found dict on 404."""
    try:
        data = get(f"deals/{deal_id}")
        d = data.get("deal", {})
        return {
            "title": d.get("title", ""),
            "group": str(d.get("group")),
            "stage": str(d.get("stage")),
            "status": str(d.get("status")),
            "value_zar": round(int(d.get("value", 0)) / 100),
        }
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {"not_found": True}
        raise


def classify(rec):
    """Classify a fetched deal record into a human-meaningful bucket."""
    if rec.get("not_found"):
        return "not_found"
    status = rec.get("status")
    group = rec.get("group")
    stage = rec.get("stage")
    if status == "2":
        return "lost"
    if status == "1":
        return "won"
    if group == "5":
        return {"43": "progressed_demo", "46": "progressed_negotiation",
                "47": "progressed_agreed"}.get(stage, "progressed_other")
    if group == "4":
        return "refreshed_new_lead"
    if group == "8":
        return "long_term_interest"
    if group == "9":
        return "partnerships"
    if group == "10":
        return "sponsorships"
    if group == "6":
        return "customer_account_mgmt"
    return f"other_group_{group}"


CLASSIFICATION_LABELS = {
    "progressed_demo": "Progressed — Demo/Pilot",
    "progressed_negotiation": "Progressed — Negotiation",
    "progressed_agreed": "Progressed — Agreed",
    "progressed_other": "Progressed — Sales Conversion",
    "refreshed_new_lead": "Refreshed — back in New Lead",
    "long_term_interest": "Long Term Interest",
    "partnerships": "Partnerships pipeline",
    "sponsorships": "Sponsorships pipeline",
    "customer_account_mgmt": "Customer Account Mgmt",
    "won": "WON",
    "lost": "Lost / Not Interested",
    "not_found": "Deleted / merged in AC",
}


def load_tracker():
    if TRACKER_FILE.exists():
        return json.loads(TRACKER_FILE.read_text(encoding="utf-8"))
    print("❌  No followup_tracker.json found. This script expects a bootstrap file to exist.")
    print("    (It was seeded once from a manual full analysis — see project notes.)")
    sys.exit(1)


def load_current_stale_ids():
    if not VELOCITY_FILE.exists():
        print("❌  velocity_data.json not found. Run update_velocity.py first.")
        sys.exit(1)
    vd = json.loads(VELOCITY_FILE.read_text(encoding="utf-8"))
    stale = vd.get("pipeline4", {}).get("stale_deals", [])
    return sorted(set(int(s["id"]) for s in stale)), vd.get("generated_at", "")


def main(full_reverify=False, no_push=False):
    print()
    print("🔄  Ear Academy — Updating Follow-Up Tracker")
    print("=" * 46)
    print(f"  ✅  Will write to:    followup_tracker.json")
    print(f"  🚫  Will NOT touch:   any HTML file")
    print()

    pull_ok = True
    if not no_push:
        print("⬇️   Pulling latest from GitHub first...")
        pull = subprocess.run(["git", "-C", str(SCRIPT_DIR), "pull", "origin", "main"],
                               capture_output=True, text=True)
        if pull.returncode != 0:
            pull_ok = False
            print(f"  ⚠️  git pull failed:\n{pull.stdout.strip()}\n{pull.stderr.strip()}")
            print("  Will still fetch fresh data and save it locally below, but will")
            print("  SKIP pushing at the end. Resolve the pull manually, then push.")
        print()

    tracker = load_tracker()
    current_stale_ids, velocity_generated_at = load_current_stale_ids()
    last_seen = set(tracker.get("last_seen_stale_ids", []))
    tracked = tracker.get("tracked", {})

    if full_reverify:
        # Re-check EVERY tracked deal (expensive — weekly sweep)
        to_check = [int(k) for k in tracked.keys()]
        print(f"🔍  Full re-verify: checking all {len(to_check)} tracked deals…")
    else:
        # Only deals that dropped off stale since the last run and aren't tracked yet
        newly_dropped = last_seen - set(current_stale_ids)
        to_check = sorted(i for i in newly_dropped if str(i) not in tracked)
        print(f"📋  Incremental: {len(newly_dropped)} deals dropped off stale since last run,")
        print(f"    {len(to_check)} are new (not already tracked) → checking those.")

    print()
    checked = 0
    for deal_id in to_check:
        rec = fetch_deal_status(deal_id)
        cls = classify(rec)
        entry = {
            **{k: v for k, v in rec.items() if k != "not_found"},
            "classification": cls,
            "last_verified": TODAY.strftime("%Y-%m-%d"),
        }
        if str(deal_id) not in tracked:
            entry["first_dropped_seen"] = TODAY.strftime("%Y-%m-%d")
        else:
            entry["first_dropped_seen"] = tracked[str(deal_id)].get("first_dropped_seen", TODAY.strftime("%Y-%m-%d"))
        tracked[str(deal_id)] = entry
        checked += 1
        print(f"    {checked}/{len(to_check)}: {deal_id} → {CLASSIFICATION_LABELS.get(cls, cls)}", end="\r")
    if to_check:
        print()
    print()

    # Summary
    from collections import Counter
    counts = Counter(v["classification"] for v in tracked.values())
    print("📊  Tracker summary (all-time, since 17 Aug cleanup start):")
    for cls, label in CLASSIFICATION_LABELS.items():
        if counts.get(cls):
            print(f"    {label}: {counts[cls]}")
    print(f"    TOTAL tracked: {len(tracked)}")
    print()

    tracker_out = {
        "schema_version": 1,
        "cleanup_window_start": tracker.get("cleanup_window_start", "2026-08-17T08:30:18Z"),
        "last_run": TODAY.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_full_reverify": TODAY.strftime("%Y-%m-%dT%H:%M:%SZ") if full_reverify else tracker.get("last_full_reverify"),
        "last_seen_stale_ids": current_stale_ids,
        "tracked": tracked,
    }
    TRACKER_FILE.write_text(json.dumps(tracker_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅  followup_tracker.json updated ({TODAY_LABEL})")
    print()

    if no_push:
        print("🛑  --no-push set — skipping git commit + push (testing mode)")
        print()
        return
    if not pull_ok:
        print("🛑  Skipping push — the pull at the start of this run failed.")
        print()
        return

    print("🚀  Publishing to GitHub…")
    try:
        os.chdir(SCRIPT_DIR)
        subprocess.run(["git", "add", "followup_tracker.json"], check=True, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", f"Update follow-up tracker — {strf(TODAY, '%-d %b %Y')}"],
            capture_output=True, text=True,
        )
        if "nothing to commit" in result.stdout:
            print("  → No changes to push (tracker unchanged since last run)")
        else:
            subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
            print("  → Pushed successfully")
        print()
        print("🎉  Done!")
        print()
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Git error: {e}")
        print("  followup_tracker.json was updated locally. Run git push manually.")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the Ear Academy stale-pipeline follow-up tracker.")
    parser.add_argument("--full-reverify", action="store_true",
                         help="Re-check every tracked deal's current status (expensive — run weekly).")
    parser.add_argument("--no-push", action="store_true",
                         help="Write followup_tracker.json locally but skip git commit + push (testing mode).")
    args = parser.parse_args()
    main(full_reverify=args.full_reverify, no_push=args.no_push)
