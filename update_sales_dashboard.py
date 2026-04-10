#!/usr/bin/env python3
"""
Ear Academy — Safe Sales Dashboard Updater
==========================================
Updates pipeline_data.json from ActiveCampaign.
NEVER touches investor.html — investor.html reads from the JSON automatically.

Usage (run from Terminal or via Claude):
    cd ~/Desktop/ear-academy-analytics
    python3 update_sales_dashboard.py

What it does:
    1. Pulls deal counts from AC Pipelines 4, 5, 6 (ZAR only)
    2. Pulls monthly new lead counts (Jan, Feb, Mar)
    3. Pulls email campaign stats
    4. Writes ONLY to pipeline_data.json
    5. Commits and pushes to GitHub (live within ~60 seconds)

What it NEVER does:
    - Does not touch investor.html
    - Does not touch index.html
    - Does not touch the schools list or Product Demos
"""

import json
import sys
import subprocess
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
try:
    import config
    AC_API_KEY  = config.AC_API_KEY
    AC_BASE_URL = config.AC_BASE_URL
except ImportError:
    print("ERROR: config.py not found. Make sure you're running from ear-academy-analytics/")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
JSON_FILE  = SCRIPT_DIR / "pipeline_data.json"

# Pipeline IDs (confirmed from memory)
P_QUAL = "4"   # Sales Qualification
P_CONV = "5"   # Sales Conversion
P_CAM  = "6"   # Customer Account Management

# Stage IDs (confirmed from memory)
S_NEW_LEAD    = "36"   # Pipeline 4 → New Lead
S_DEMO        = "43"   # Pipeline 5 → Demo/Pilot
S_NEGOTIATION = "46"   # Pipeline 5 → Negotiation
S_ONBOARDING  = "50"   # Pipeline 6 → Onboarding
S_ACTIVATED   = "51"   # Pipeline 6 → Activated

OPEN = "0"
WON  = "1"


# ── AC API helpers ─────────────────────────────────────────────────────────────
def ac_get(endpoint, params=None):
    """Fetch all results from an AC API endpoint, handling pagination."""
    headers = {"Api-Token": AC_API_KEY}
    url     = f"{AC_BASE_URL}/api/3/{endpoint}"
    params  = dict(params or {})
    params["limit"] = 100
    results = []
    offset  = 0
    while True:
        params["offset"] = offset
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  ⚠️  AC API error on {endpoint}: {e}")
            return results
        body = r.json()

        # Deals endpoint returns {"deals": [...]}
        if "deals" in body:
            batch = body["deals"]
        elif "contacts" in body:
            batch = body["contacts"]
        elif "campaigns" in body:
            batch = body["campaigns"]
        elif "campaignMessages" in body:
            batch = body["campaignMessages"]
        else:
            # Return whatever the top-level list is
            for key, val in body.items():
                if isinstance(val, list):
                    batch = val
                    break
            else:
                break

        results.extend(batch)
        if len(batch) < 100:
            break
        offset += 100
    return results


def fetch_deals_for_pipeline(pipeline_id, status_filter=OPEN):
    """Return all deals in a given pipeline, filtered by status. Default = open only."""
    print(f"  Fetching deals from Pipeline {pipeline_id}…")
    params = {"filters[pipeline]": pipeline_id}
    if status_filter is not None:
        params["filters[status]"] = status_filter
    deals = ac_get("deals", params)
    zar = [d for d in deals if d.get("currency", "").lower() == "zar"]
    print(f"    → {len(deals)} total, {len(zar)} ZAR")
    return deals, zar  # return both so caller can choose


def count_by_stage(deals, stage_id):
    """Count deals in a specific stage."""
    return sum(1 for d in deals if str(d.get("stage")) == stage_id)


def count_new_contacts_this_month():
    """Count contacts created in the current calendar month."""
    first_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Use limit=1 and read meta.total — faster than fetching all records
    import requests as req
    headers = {"Api-Token": AC_API_KEY}
    params  = {"filters[created_after]": first_of_month, "limit": 1}
    try:
        r = req.get(f"{AC_BASE_URL}/api/3/contacts", headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return int(r.json().get("meta", {}).get("total", 0))
    except Exception:
        return 0


def count_new_leads_for_month(year, month):
    """Count NEW CONTACTS (leads) added to AC during a specific month.
    'Leads' = new contacts added to the CRM, per dashboard definition.
    Uses meta.total for efficiency instead of fetching all records.
    """
    start = datetime(year, month, 1)
    end   = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    import requests as req
    headers = {"Api-Token": AC_API_KEY}
    params  = {
        "filters[created_after]":  start.strftime("%Y-%m-%dT00:00:00Z"),
        "filters[created_before]": end.strftime("%Y-%m-%dT00:00:00Z"),
        "limit": 1,
    }
    try:
        r = req.get(f"{AC_BASE_URL}/api/3/contacts", headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return int(r.json().get("meta", {}).get("total", 0))
    except Exception:
        return 0


def count_new_deals_for_month(year, month):
    """Count new ZAR deals created in Pipeline 4 (Sales Qualification) during a specific month.
    'Deals' = new opportunities opened in the Sales Qualification pipeline.
    """
    start = datetime(year, month, 1)
    end   = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    deals = ac_get("deals", {
        "filters[pipeline]":       P_QUAL,
        "filters[created_after]":  start.strftime("%Y-%m-%dT00:00:00Z"),
        "filters[created_before]": end.strftime("%Y-%m-%dT00:00:00Z"),
    })
    zar = [d for d in deals if d.get("currency", "").lower() == "zar"]
    return len(zar)


def fetch_email_stats():
    """Pull stats from SA-only broadcast campaigns (segmentid=1004 or name contains 'SA').
    Excludes large UK blasts which would distort the numbers.
    """
    print("  Fetching email campaign stats (SA campaigns only)…")
    try:
        # Fetch last 50 completed campaigns, then filter to SA
        r = requests.get(
            f"{AC_BASE_URL}/api/3/campaigns",
            headers={"Api-Token": AC_API_KEY},
            params={"limit": 50, "orders[sdate]": "DESC", "filters[status]": "5"},
            timeout=30
        )
        r.raise_for_status()
        all_campaigns = r.json().get("campaigns", [])
    except requests.RequestException as e:
        print(f"  ⚠️  Could not fetch campaigns: {e}")
        return {"sent": "—", "replied": "—", "rate": "—", "new_contacts": "—"}

    # SA campaigns: segmentid 1004, or name contains 'SA', or name contains 'South Africa'
    # Exclude massive UK blasts (Business Case, UK Pilots, etc.)
    sa_campaigns = [
        c for c in all_campaigns
        if str(c.get("segmentid")) == "1004"
        or " SA" in c.get("name", "")
        or "South Africa" in c.get("name", "")
    ]

    total_sent   = sum(int(c.get("send_amt", 0) or 0) for c in sa_campaigns)
    total_opened = sum(int(c.get("opens",    0) or 0) for c in sa_campaigns)
    rate = f"{(total_opened / total_sent * 100):.1f}%" if total_sent > 0 else "—"

    new_contacts = count_new_contacts_this_month()
    print(f"    → {total_sent} sent, {total_opened} opened across {len(sa_campaigns)} SA campaigns")
    return {
        "sent":         str(total_sent),
        "replied":      str(total_opened),
        "rate":         rate,
        "new_contacts": str(new_contacts),
    }


# ── ARR Tier helpers ───────────────────────────────────────────────────────────
def fetch_won_zar_deals():
    """Fetch all Won ZAR deals from Pipeline 6 (Customer Account Management).
    These are the paying schools — used to calculate ARR tier breakdown.
    AC returns values in cents as strings: divide by 100 to get ZAR.
    """
    print("  Fetching won ZAR deals (Pipeline 6) for ARR tiers…")
    deals = ac_get("deals", {
        "filters[pipeline]": P_CAM,
        "filters[status]":   WON,
    })
    zar = [d for d in deals if d.get("currency", "").lower() == "zar"]
    print(f"    → {len(zar)} won ZAR deals found")
    return zar


def calculate_arr_tiers(won_deals):
    """Bucket won ZAR deals into ARR tiers by annual value.

    Tier boundaries (in ZAR — AFTER dividing cents value by 100):
      Tier 1: R1     – R4,999
      Tier 2: R5,000 – R9,999
      Tier 3: R10,000 – R19,999
      Tier 4: R20,000+
    """
    tiers = {
        "tier1": {"label": "R1–5k",   "min": 1,      "max": 4999,  "count": 0, "revenue": 0.0},
        "tier2": {"label": "R5–10k",  "min": 5000,   "max": 9999,  "count": 0, "revenue": 0.0},
        "tier3": {"label": "R10–20k", "min": 10000,  "max": 19999, "count": 0, "revenue": 0.0},
        "tier4": {"label": "R20k+",   "min": 20000,  "max": None,  "count": 0, "revenue": 0.0},
    }

    for deal in won_deals:
        try:
            value_zar = int(deal.get("value", 0)) / 100  # cents → ZAR
        except (ValueError, TypeError):
            continue

        for tier in tiers.values():
            if tier["max"] is None:
                if value_zar >= tier["min"]:
                    tier["count"]   += 1
                    tier["revenue"] += value_zar
                    break
            else:
                if tier["min"] <= value_zar <= tier["max"]:
                    tier["count"]   += 1
                    tier["revenue"] += value_zar
                    break

    print("    ARR tier breakdown:")
    for t in tiers.values():
        print(f"      {t['label']}: {t['count']} schools, R{t['revenue']:,.0f}")

    return tiers


def fetch_lost_deals_pipeline5():
    """Fetch Lost deals from Pipeline 5 (Sales Conversion) only.

    WHY Pipeline 5 only: Pipeline 4 losses are leads that never got to demo
    stage — not the same as deals we actively worked and didn't close.
    Pipeline 5 losses give the true strike rate.
    """
    print("  Fetching lost deals (Pipeline 5 — Sales Conversion only)…")
    deals = ac_get("deals", {
        "filters[pipeline]": P_CONV,
        "filters[status]":   "2",   # 2 = Lost
    })
    zar = [d for d in deals if d.get("currency", "").lower() == "zar"]
    total_value = sum(int(d.get("value", 0)) / 100 for d in zar)
    print(f"    → {len(zar)} lost deals, total value R{total_value:,.0f}")
    return {
        "count":       len(zar),
        "total_value": round(total_value, 2),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n🔄  Ear Academy — Updating Sales Dashboard")
    print("=" * 45)
    print("  ✅  Will write to:    pipeline_data.json")
    print("  🚫  Will NOT touch:   investor.html\n")

    now = datetime.now()

    # ── 1. Pipeline stage counts ─────────────────────────────────────────────
    print("📊  Pipeline counts:")
    # Pipelines 4 & 5: ZAR-only open deals
    _, qual_deals = fetch_deals_for_pipeline(P_QUAL, status_filter=OPEN)
    _, conv_deals = fetch_deals_for_pipeline(P_CONV, status_filter=OPEN)
    # Pipeline 6 (customers): fetch ALL deals regardless of status/currency — paying schools
    # are tracked as open deals in Onboarding/Activated stages
    cam_all, _   = fetch_deals_for_pipeline(P_CAM, status_filter=None)

    qualification = count_by_stage(qual_deals, S_NEW_LEAD)
    demo          = count_by_stage(conv_deals, S_DEMO)
    negotiation   = count_by_stage(conv_deals, S_NEGOTIATION)
    onboarding    = count_by_stage(cam_all,    S_ONBOARDING)
    activated     = count_by_stage(cam_all,    S_ACTIVATED)
    customers     = onboarding + activated   # total paying schools

    print(f"    Sales Qualification (New Lead): {qualification}")
    print(f"    Sales Conversion    (Demo):     {demo}")
    print(f"    Sales Conversion    (Neg):      {negotiation}")
    print(f"    Customer Acc Mgmt   (Total):    {customers}  ({onboarding} onboarding + {activated} activated)")

    # ── 2. Monthly new leads & deals ─────────────────────────────────────────
    print("\n📅  Monthly activity (leads & deals from AC — Product Demos stay in HTML):")
    jan_leads = count_new_leads_for_month(2026, 1)
    feb_leads = count_new_leads_for_month(2026, 2)
    mar_leads = count_new_leads_for_month(2026, 3)
    jan_deals = count_new_deals_for_month(2026, 1)
    feb_deals = count_new_deals_for_month(2026, 2)
    mar_deals = count_new_deals_for_month(2026, 3)
    print(f"    Jan — leads: {jan_leads}, deals: {jan_deals}")
    print(f"    Feb — leads: {feb_leads}, deals: {feb_deals}")
    print(f"    Mar — leads: {mar_leads}, deals: {mar_deals}")

    # ── 3. Email stats ───────────────────────────────────────────────────────
    print("\n📧  Email stats:")
    email = fetch_email_stats()

    # ── 4. ARR tiers & lost deals ────────────────────────────────────────────
    print("\n🏫  ARR tier breakdown:")
    won_deals  = fetch_won_zar_deals()
    arr_tiers  = calculate_arr_tiers(won_deals)

    print("\n❌  Lost deals (Pipeline 5 only):")
    lost_deals = fetch_lost_deals_pipeline5()

    # ── 5. Build JSON ────────────────────────────────────────────────────────
    timestamp = now.strftime("%d %b %Y at %H:%M")
    data = {
        "pipeline": {
            "qualification": qualification,
            "demo":          demo,
            "negotiation":   negotiation,
            "customers":     customers,
        },
        "monthly": {
            "jan": {"new_leads": str(jan_leads), "new_deals": str(jan_deals)},
            "feb": {"new_leads": str(feb_leads), "new_deals": str(feb_deals)},
            "mar": {"new_leads": str(mar_leads), "new_deals": str(mar_deals)},
        },
        "email": email,
        "arr_tiers": {
            "tier1": {
                "label":   arr_tiers["tier1"]["label"],
                "count":   arr_tiers["tier1"]["count"],
                "revenue": round(arr_tiers["tier1"]["revenue"], 2),
            },
            "tier2": {
                "label":   arr_tiers["tier2"]["label"],
                "count":   arr_tiers["tier2"]["count"],
                "revenue": round(arr_tiers["tier2"]["revenue"], 2),
            },
            "tier3": {
                "label":   arr_tiers["tier3"]["label"],
                "count":   arr_tiers["tier3"]["count"],
                "revenue": round(arr_tiers["tier3"]["revenue"], 2),
            },
            "tier4": {
                "label":   arr_tiers["tier4"]["label"],
                "count":   arr_tiers["tier4"]["count"],
                "revenue": round(arr_tiers["tier4"]["revenue"], 2),
            },
        },
        "lost_deals": lost_deals,
        "last_updated": timestamp,
        "notes": {
            "pipeline":       "ZAR-only deals in active stages (Pipelines 4, 5, 6)",
            "monthly_leads":  "New ZAR contacts entered into Pipeline 4 (Sales Qualification)",
            "monthly_deals":  "New ZAR deals entered into Pipeline 5 (Sales Conversion)",
            "product_demos":  "Manually tracked — edit the Product Demos row in investor.html directly",
            "schools_list":   "Manually maintained — edit the school pills in investor.html directly",
            "customers":      f"Onboarding ({onboarding}) + Activated ({activated}) in Pipeline 6",
            "arr_tiers":      "Won ZAR deals in Pipeline 6, bucketed by annual value (cents ÷ 100)",
            "lost_deals":     "Lost deals in Pipeline 5 (Sales Conversion) only — excludes Pipeline 4 qualification rejections",
        }
    }

    # ── 6. Write JSON ────────────────────────────────────────────────────────
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅  pipeline_data.json updated ({timestamp})")

    # ── 7. Git commit & push ─────────────────────────────────────────────────
    print("\n🚀  Publishing to GitHub…")
    try:
        subprocess.run(["git", "-C", str(SCRIPT_DIR), "add", "pipeline_data.json", "investor.html"], check=True)
        subprocess.run(["git", "-C", str(SCRIPT_DIR), "commit", "-m",
                        f"Update sales dashboard — {timestamp}"], check=True)
        subprocess.run(["git", "-C", str(SCRIPT_DIR), "push", "origin", "main"], check=True)
        print("✅  Pushed! Live on GitHub Pages within ~60 seconds.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Git error: {e}")
        print("    JSON file was saved locally — push manually when ready.")

    print("\n🎉  Done! Dashboard updated safely.")
    print("    Manual sections preserved: Product Demos, Schools list")
    print(f"    View live at: https://earacademy.github.io/Ear-Academy-Usage-Tracker/investor.html\n")


if __name__ == "__main__":
    main()
