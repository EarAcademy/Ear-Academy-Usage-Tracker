#!/usr/bin/env python3
"""
Ear Academy — Pipeline Velocity Updater
========================================
Pulls live deal data from ActiveCampaign for all three pipeline stages
and writes velocity_data.json. The pipeline_velocity.html page reads
this JSON on load.

SAFE BY DESIGN:
  ✅  Writes ONLY to velocity_data.json
  🚫  Never touches pipeline_velocity.html or any other file

Run weekly alongside update_sales_dashboard.py:
  cd ~/Desktop/ear-academy-analytics && python3 update_velocity.py && git add -A && git commit -m "Update velocity $(date '+%Y-%m-%d')" && git push origin main
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Load credentials from config.py ─────────────────────────────────────────
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

OUTPUT_FILE = SCRIPT_DIR / "velocity_data.json"
TODAY = datetime.now(timezone.utc)
TODAY_LABEL = TODAY.strftime("%-d %b %Y at %H:%M")

# ── Pipeline / Stage constants (from project docs) ───────────────────────────
PIPELINE_4 = 4   # Sales Qualification
PIPELINE_5 = 5   # Sales Conversion
STAGE_NEW_LEAD   = 36
STAGE_DEMO       = 43
STAGE_NEGO       = 46

# Stage name lookup (covers historical IDs that appear in activity logs)
STAGE_NAMES = {
    "36": "New Lead",
    "43": "Demo/Pilot", "44": "Demo/Pilot", "45": "Demo/Pilot",
    "46": "Negotiation", "47": "Negotiation",
    "13": "Demo/Pilot", "26": "Demo/Pilot",
    "27": "Negotiation",
    "25": "Onboarding", "50": "Onboarding", "51": "Activated",
}

HEADERS = {
    "Api-Token": AC_API_KEY,
    "Content-Type": "application/json",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get(endpoint, params=None):
    """Simple GET wrapper with error handling."""
    url = f"{AC_BASE_URL.rstrip('/')}/api/3/{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_all_deals(pipeline_id):
    """Paginate through all deals in a pipeline, return ZAR open deals only."""
    deals = []
    offset = 0
    while True:
        data = get("deals", {
            "filters[group]": pipeline_id,
            "filters[status]": 0,
            "limit": 100,
            "offset": offset,
        })
        batch = data.get("deals", [])
        deals.extend(batch)
        total = int(data.get("meta", {}).get("total", 0))
        offset += len(batch)
        if offset >= total or not batch:
            break

    zar = [d for d in deals if d.get("currency", "").lower() == "zar"]
    return zar


def days_since(date_str):
    """Return integer days between a date string and today."""
    if not date_str:
        return None
    try:
        s = date_str.replace(" ", "T")
        # Strip AC timezone offsets so fromisoformat works on Python 3.9
        for tz in ["-06:00", "-05:00", "+00:00"]:
            s = s.replace(tz, "")
        s = s.rstrip("Z")
        d = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return max(0, (TODAY - d).days)
    except Exception:
        return None


def bucket_deals(deals):
    """Return time-in-stage bucket counts for a list of deals."""
    b = {"u30": 0, "u60": 0, "u90": 0, "u180": 0, "over180": 0}
    for d in deals:
        n = d.get("days_in_stage", 0) or 0
        if   n <= 30:  b["u30"]    += 1
        elif n <= 60:  b["u60"]    += 1
        elif n <= 90:  b["u90"]    += 1
        elif n <= 180: b["u180"]   += 1
        else:          b["over180"] += 1
    return b


def enrich_deal(deal):
    """Add computed fields to a deal dict."""
    edate = deal.get("edate") or deal.get("cdate")
    cdate = deal.get("cdate")
    deal["days_in_stage"] = days_since(edate)
    deal["total_age_days"] = days_since(cdate)
    deal["value_zar"] = round(int(deal.get("value", 0)) / 100)
    return deal


def fetch_stage_history(deal_id):
    """
    Pull deal activity log and reconstruct stage move history.
    Returns list of {stage_name, stage_id, entered_date, days} in order.
    """
    try:
        data = get(f"deals/{deal_id}/dealActivities", {"limit": 200})
    except Exception:
        return []

    activities = data.get("dealActivities", [])

    # Filter to stage-change events: dataType == "d_stageid"
    moves = []
    for a in activities:
        if a.get("dataType") == "d_stageid":
            moves.append({
                "from_stage": a.get("dataOldval", ""),
                "to_stage":   a.get("dataAction", ""),
                "date":       a.get("cdate", ""),
            })

    # Sort chronologically
    moves.sort(key=lambda x: x["date"])

    if not moves:
        return []

    # Build timeline: each entry is (stage_id, entered_date)
    timeline = []
    for i, move in enumerate(moves):
        stage_id = move["to_stage"]
        entered  = move["date"]
        if i + 1 < len(moves):
            exited = moves[i + 1]["date"]
            d_in   = days_since(exited) # approximate — days from exit to today, minus days from entry to today
            # More accurate: parse both dates
            try:
                def parse(s):
                    s = s.replace(" ","T")
                    for tz in ["-06:00","-05:00","+00:00"]: s=s.replace(tz,"")
                    return datetime.fromisoformat(s.rstrip("Z")).replace(tzinfo=timezone.utc)
                d_in = max(0, (parse(exited) - parse(entered)).days)
            except Exception:
                d_in = None
        else:
            # Still in this stage
            d_in = days_since(entered)

        stage_name = STAGE_NAMES.get(str(stage_id), f"Stage {stage_id}")
        # Only include stages we care about
        if stage_name in ("New Lead", "Demo/Pilot", "Negotiation", "Onboarding", "Activated"):
            timeline.append({
                "stage_id":   stage_id,
                "stage_name": stage_name,
                "entered":    entered[:10],
                "days":       d_in,
            })

    return timeline


# ── Main fetch functions ───────────────────────────────────────────────────────

def fetch_pipeline4():
    """All ZAR open deals in Pipeline 4 (New Lead stage)."""
    print("  Fetching Pipeline 4 (New Lead)…")
    deals = fetch_all_deals(PIPELINE_4)
    deals = [enrich_deal(d) for d in deals]
    buckets = bucket_deals(deals)
    avg = round(sum(d["days_in_stage"] or 0 for d in deals) / len(deals)) if deals else 0
    total_val = sum(d["value_zar"] for d in deals)
    print(f"    → {len(deals)} ZAR open deals")
    print(f"    → Avg {avg} days in stage")
    print(f"    → Buckets: 0-30d={buckets['u30']}, 31-60d={buckets['u60']}, "
          f"61-90d={buckets['u90']}, 91-180d={buckets['u180']}, 180d+={buckets['over180']}")
    return {
        "total": len(deals),
        "avg_days_in_stage": avg,
        "total_value_zar": total_val,
        "buckets": buckets,
    }


def fetch_pipeline5():
    """All ZAR open deals in Pipeline 5 split by stage, with deal-level detail."""
    print("  Fetching Pipeline 5 (Sales Conversion)…")
    deals = fetch_all_deals(PIPELINE_5)
    deals = [enrich_deal(d) for d in deals]

    demo = [d for d in deals if str(d.get("stage")) == str(STAGE_DEMO)]
    nego = [d for d in deals if str(d.get("stage")) == str(STAGE_NEGO)]

    def summarise(deal_list):
        avg = round(sum(d["days_in_stage"] or 0 for d in deal_list) / len(deal_list)) if deal_list else 0
        val = sum(d["value_zar"] for d in deal_list)
        buckets = bucket_deals(deal_list)
        detail = [{
            "id":             d.get("id"),
            "title":          d.get("title", ""),
            "stage":          str(d.get("stage")),
            "days_in_stage":  d.get("days_in_stage"),
            "total_age_days": d.get("total_age_days"),
            "value_zar":      d.get("value_zar"),
            "edate":          (d.get("edate") or "")[:10],
            "cdate":          (d.get("cdate") or "")[:10],
        } for d in sorted(deal_list, key=lambda x: -(x.get("days_in_stage") or 0))]
        return {"count": len(deal_list), "avg_days_in_stage": avg,
                "total_value_zar": val, "buckets": buckets, "deals": detail}

    demo_data = summarise(demo)
    nego_data = summarise(nego)

    print(f"    → Demo/Pilot:   {demo_data['count']} deals, avg {demo_data['avg_days_in_stage']}d in stage")
    print(f"    → Negotiation:  {nego_data['count']} deals, avg {nego_data['avg_days_in_stage']}d in stage")

    # Conversion rates
    total_in_conv = demo_data["count"] + nego_data["count"]
    p4_total = None  # filled in by caller
    return {
        "demo":  demo_data,
        "nego":  nego_data,
        "total_in_conversion": total_in_conv,
    }


def fetch_stage_histories(p5_data):
    """
    For each deal in Demo and Negotiation, fetch full stage history.
    This is the most time-consuming step — one API call per deal.
    """
    all_deals = p5_data["demo"]["deals"] + p5_data["nego"]["deals"]
    total = len(all_deals)
    print(f"  Fetching stage histories for {total} conversion deals…")
    for i, deal in enumerate(all_deals, 1):
        deal["stage_history"] = fetch_stage_history(deal["id"])
        print(f"    {i}/{total}: {deal['title'][:45]}", end="\r")
    print()  # newline after progress
    return p5_data


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("🔄  Ear Academy — Updating Pipeline Velocity")
    print("=" * 46)
    print(f"  ✅  Will write to:    velocity_data.json")
    print(f"  🚫  Will NOT touch:   pipeline_velocity.html")
    print()

    # ── 1. Pipeline 4 (New Leads) ────────────────────────────────────────────
    print("📊  Pipeline 4 — New Lead stage:")
    p4 = fetch_pipeline4()
    print()

    # ── 2. Pipeline 5 (Sales Conversion) ────────────────────────────────────
    print("📊  Pipeline 5 — Sales Conversion:")
    p5 = fetch_pipeline5()
    print()

    # ── 3. Stage histories for conversion deals ──────────────────────────────
    print("🔍  Stage histories:")
    p5 = fetch_stage_histories(p5)
    print()

    # ── 4. Conversion rates ──────────────────────────────────────────────────
    total_in_conv = p5["total_in_conversion"]
    lead_to_demo_pct = round(total_in_conv / p4["total"] * 100, 1) if p4["total"] else 0
    demo_to_nego_pct = round(
        p5["nego"]["count"] / (p5["demo"]["count"] + p5["nego"]["count"]) * 100
    ) if (p5["demo"]["count"] + p5["nego"]["count"]) > 0 else 0

    print("📈  Conversion rates:")
    print(f"    Lead → Demo/Nego:  {lead_to_demo_pct}% ({total_in_conv} of {p4['total']} leads)")
    print(f"    Demo → Negotiation: {demo_to_nego_pct}%")
    print()

    # ── 5. Stuck deal counts (for hero metrics) ──────────────────────────────
    all_conv_deals = p5["demo"]["deals"] + p5["nego"]["deals"]
    stuck_60  = sum(1 for d in all_conv_deals if (d.get("days_in_stage") or 0) > 60)
    stuck_90  = sum(1 for d in all_conv_deals if (d.get("days_in_stage") or 0) > 90)
    p4_stuck_180 = p4["buckets"]["over180"]
    total_stuck_90 = stuck_90 + p4["buckets"]["u180"] + p4["buckets"]["over180"]

    print("⚠️   Stuck deal summary:")
    print(f"    Conversion deals >60d:  {stuck_60}")
    print(f"    Conversion deals >90d:  {stuck_90}")
    print(f"    New Leads >180d:        {p4_stuck_180}")
    print(f"    All stages >90d total:  {total_stuck_90}")
    print()

    # ── 6. Build output JSON ─────────────────────────────────────────────────
    output = {
        "generated_at": TODAY.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_label": TODAY_LABEL,
        "pipeline4": p4,
        "pipeline5": p5,
        "conversion_rates": {
            "lead_to_demo_pct": lead_to_demo_pct,
            "demo_to_nego_pct": demo_to_nego_pct,
        },
        "stuck_summary": {
            "conversion_over_60":  stuck_60,
            "conversion_over_90":  stuck_90,
            "new_leads_over_180":  p4_stuck_180,
            "all_stages_over_90":  total_stuck_90,
        },
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅  velocity_data.json updated ({TODAY_LABEL})")
    print()

    # ── 7. Git push ──────────────────────────────────────────────────────────
    print("🚀  Publishing to GitHub…")
    try:
        os.chdir(SCRIPT_DIR)
        subprocess.run(["git", "add", "velocity_data.json"], check=True, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", f"Update velocity data — {TODAY.strftime('%-d %b %Y')}"],
            capture_output=True, text=True,
        )
        if "nothing to commit" in result.stdout:
            print("  → No changes to push (data unchanged since last run)")
        else:
            subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
            print("  → Pushed successfully")
        print()
        print("🎉  Done! velocity_data.json is live.")
        print()
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Git error: {e}")
        print("  velocity_data.json was updated locally. Run git push manually.")
        print()


if __name__ == "__main__":
    main()
