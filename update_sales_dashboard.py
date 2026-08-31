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

import argparse
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

SCRIPT_DIR    = Path(__file__).parent
JSON_FILE     = SCRIPT_DIR / "pipeline_data.json"
ROSTER_FILE   = SCRIPT_DIR / "paying_schools.json"   # canonical paying-schools roster
RENEWALS_FILE = SCRIPT_DIR / "renewals_data.json"    # confirmed-renewals list

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
S_UPCOMING_RENEWAL = "52"   # Pipeline 6 → Upcoming Renewal (still a paying customer)
S_RENEWED          = "70"   # Pipeline 6 → Renewed (still a paying customer)

# Paying-customer stages: any deal in one of these still counts as an active
# paying school. Excludes Low Activity (53), Churning (54), Lost (55) — those
# are genuinely at-risk/gone, not just mid-renewal-cycle.
PAYING_STAGES = (S_ONBOARDING, S_ACTIVATED, S_UPCOMING_RENEWAL, S_RENEWED)

OPEN = "0"
WON  = "1"


def _is_test_deal(title):
    """True for automation/debug deals (e.g. the renewal-automation Zap's
    test fixtures), which must never be counted as real paying schools or
    real revenue. Matches the 'ZZZ ...' naming convention used for them."""
    return (title or "").strip().upper().startswith("ZZZ")


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
        if _is_test_deal(deal.get("title")):
            continue
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


# ── Paying-schools roster helpers ─────────────────────────────────────────────
def fetch_account_names(account_ids):
    """Look up account display names from AC for a set of account IDs.
    Returns { account_id (str): account_name (str) }. Missing/failed → ''.
    """
    if not account_ids:
        return {}
    headers = {"Api-Token": AC_API_KEY}
    names   = {}
    for aid in account_ids:
        if not aid:
            continue
        try:
            r = requests.get(
                f"{AC_BASE_URL}/api/3/accounts/{aid}",
                headers=headers, timeout=30,
            )
            r.raise_for_status()
            names[str(aid)] = (r.json().get("account") or {}).get("name", "").strip()
        except requests.RequestException:
            names[str(aid)] = ""
    return names


def build_paying_schools_roster(cam_deals):
    """Build the canonical list of paying schools from Pipeline 6 deals.

    Filters (per system policy):
      • currency == ZAR
      • stage in PAYING_STAGES (Onboarding, Activated, Upcoming Renewal, Renewed)
      • exclude B2C accounts (deal title or account name contains 'B2C',
        case-insensitive) — these are individual subscribers, not schools.
      • exclude automation test/debug deals (title starts with 'ZZZ')

    Returns a list of dicts with deal + account info. Sorted by title.
    """
    matching = []
    for d in cam_deals:
        if (d.get("currency") or "").lower() != "zar":
            continue
        if str(d.get("stage")) not in PAYING_STAGES:
            continue
        title = (d.get("title") or "").strip()
        if "b2c" in title.lower() or _is_test_deal(title):
            continue
        matching.append(d)

    # Look up account names so the roster has both AC-side identifiers.
    account_ids   = {str(d.get("account")) for d in matching if d.get("account")}
    account_names = fetch_account_names(account_ids)

    roster = []
    for d in matching:
        aid   = str(d.get("account")) if d.get("account") else None
        aname = account_names.get(aid, "") if aid else ""
        # Second B2C check — sometimes the account name carries the marker.
        if aname and "b2c" in aname.lower():
            continue

        title = (d.get("title") or "").strip()
        try:
            value_zar = int(d.get("value", 0)) / 100   # cents → ZAR
        except (ValueError, TypeError):
            value_zar = 0

        stage_id    = str(d.get("stage"))
        stage_label = ("Activated"         if stage_id == S_ACTIVATED         else
                       "Onboarding"        if stage_id == S_ONBOARDING        else
                       "Upcoming Renewal"  if stage_id == S_UPCOMING_RENEWAL  else
                       "Renewed"           if stage_id == S_RENEWED           else
                       stage_id)

        roster.append({
            "deal_id":      str(d.get("id", "")),
            "title":        title,
            "stage":        stage_label,
            "stage_id":     stage_id,
            "value_zar":    int(value_zar),
            "account_id":   aid,
            "account_name": aname or None,
        })

    roster.sort(key=lambda r: r["title"].lower())
    return roster


# TEMPORARY (added 2026-08-31, Brandon's request): the renewal automation has no
# real "Renewed" deals yet, so show its "ZZZ TEST..." fixtures as placeholder
# data to demo the page. Flip back to False once real renewals start landing --
# ZZZ-test deals must NEVER count in the roster/ARR/customer numbers (those
# still exclude them via _is_test_deal(), unaffected by this flag).
RENEWALS_INCLUDE_TEST_DEALS = True


def _build_stage_deal_list(cam_all, stage_id, date_field):
    """Shared filter/shape logic for a single Pipeline 6 stage's deal list.
    Same currency/B2C exclusions as the paying-schools roster. Test-deal
    exclusion is gated by RENEWALS_INCLUDE_TEST_DEALS -- every OTHER
    consumer of _is_test_deal() (roster, ARR tiers, customer count) always
    excludes them regardless of this flag.

    `date_field` names the output date key (e.g. 'renewed_on') and is
    filled from the deal's last-modified date -- AC doesn't expose a
    stage-entry timestamp directly, so this is a proxy for "since when
    has this deal been in this stage".
    """
    matching = []
    for d in cam_all:
        if (d.get("currency") or "").lower() != "zar":
            continue
        if str(d.get("stage")) != stage_id:
            continue
        title = (d.get("title") or "").strip()
        if "b2c" in title.lower():
            continue
        if _is_test_deal(title) and not RENEWALS_INCLUDE_TEST_DEALS:
            continue
        matching.append(d)

    account_ids   = {str(d.get("account")) for d in matching if d.get("account")}
    account_names = fetch_account_names(account_ids)

    out = []
    for d in matching:
        aid   = str(d.get("account")) if d.get("account") else None
        aname = account_names.get(aid, "") if aid else ""
        if aname and "b2c" in aname.lower():
            continue

        title = (d.get("title") or "").strip()
        try:
            value_zar = int(d.get("value", 0)) / 100
        except (ValueError, TypeError):
            value_zar = 0

        out.append({
            "deal_id":      str(d.get("id", "")),
            "title":        title,
            "account_id":   aid,
            "account_name": aname or None,
            "value_zar":    int(value_zar),
            "is_test":      _is_test_deal(title),
            date_field:     (d.get("mdate") or "")[:10] or None,
        })

    out.sort(key=lambda r: r[date_field] or "", reverse=True)
    return out


def build_renewals_data(cam_all):
    """Confirmed renewals: Pipeline 6 deals in the 'Renewed' stage ONLY --
    the stage that confirms a school's renewal actually went through (not
    'Upcoming Renewal', which just means the cycle is approaching)."""
    return _build_stage_deal_list(cam_all, S_RENEWED, "renewed_on")


def build_upcoming_renewals_data(cam_all):
    """Schools due for renewal: Pipeline 6 deals in 'Upcoming Renewal' --
    the cycle is approaching but NOT yet confirmed. 'since' is when the
    deal was last modified (proxy for when it entered this stage), not an
    actual renewal due-date -- AC doesn't expose one on these deals."""
    return _build_stage_deal_list(cam_all, S_UPCOMING_RENEWAL, "since")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(no_push=False):
    print("\n🔄  Ear Academy — Updating Sales Dashboard")
    print("=" * 45)
    print("  ✅  Will write to:    pipeline_data.json + paying_schools.json")
    print("  🚫  Will NOT touch:   investor.html")
    if no_push:
        print("  🛑  --no-push: skipping git commit + push (testing mode)")
    print()

    # ── 0. Pull latest FIRST, before writing anything ────────────────────────
    # More than one operator can push to this repo now. Pulling here — before
    # this run's fresh JSON is written — means the working tree is caught up
    # first, so the commit this run makes can push as a clean fast-forward
    # instead of getting rejected (non-fast-forward) at the very end, after
    # all the AC API work has already been done.
    pull_ok = True
    if not no_push:
        print("⬇️   Pulling latest from GitHub first...")
        pull = subprocess.run(["git", "-C", str(SCRIPT_DIR), "pull", "origin", "main"],
                               capture_output=True, text=True)
        if pull.returncode != 0:
            pull_ok = False
            print(f"  ⚠️  git pull failed:\n{pull.stdout.strip()}\n{pull.stderr.strip()}")
            print("  Will still fetch fresh data and save it locally below, but will")
            print("  SKIP pushing at the end (a push would just be rejected the same")
            print("  way). Resolve the pull manually, then push when ready.")
        print()

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
    upcoming_renewal = count_by_stage(cam_all, S_UPCOMING_RENEWAL)
    renewed           = count_by_stage(cam_all, S_RENEWED)
    customers     = onboarding + activated + upcoming_renewal + renewed   # total paying schools

    print(f"    Sales Qualification (New Lead): {qualification}")
    print(f"    Sales Conversion    (Demo):     {demo}")
    print(f"    Sales Conversion    (Neg):      {negotiation}")
    print(f"    Customer Acc Mgmt   (Total):    {customers}  "
          f"({onboarding} onboarding + {activated} activated + "
          f"{upcoming_renewal} upcoming renewal + {renewed} renewed)")

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

    # ── 4b. Paying-schools roster (canonical list for usage dashboard) ───────
    print("\n🏫  Building paying-schools roster "
          "(Pipeline 6, stages 50/51/52/70, ZAR, no B2C, no test deals)…")
    roster = build_paying_schools_roster(cam_all)
    print(f"    → {len(roster)} paying schools in roster")

    # The roster applies the B2C + test-deal exclusions that the raw stage
    # count above doesn't — use it as the customer count so pipeline_data.json
    # (and investor.html's headline number) match the usage dashboard exactly.
    customers = len(roster)

    # ── 4c. Confirmed renewals (Renewed stage only) ──────────────────────────
    print("\n🔁  Building renewals list (Pipeline 6, stage 70 'Renewed' only)…")
    renewals = build_renewals_data(cam_all)
    print(f"    → {len(renewals)} confirmed renewals")

    print("⏳  Building upcoming-renewals list (Pipeline 6, stage 52 'Upcoming Renewal')…")
    upcoming_renewals = build_upcoming_renewals_data(cam_all)
    print(f"    → {len(upcoming_renewals)} due for renewal")

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
            "customers":      f"Onboarding ({onboarding}) + Activated ({activated}) + "
                               f"Upcoming Renewal ({upcoming_renewal}) + Renewed ({renewed}) in "
                               f"Pipeline 6, minus B2C/test-deal exclusions (= paying-schools roster count)",
            "arr_tiers":      "Won ZAR deals in Pipeline 6, bucketed by annual value (cents ÷ 100), excludes automation test deals",
            "lost_deals":     "Lost deals in Pipeline 5 (Sales Conversion) only — excludes Pipeline 4 qualification rejections",
        }
    }

    # ── 6. Write JSON ────────────────────────────────────────────────────────
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅  pipeline_data.json updated ({timestamp})")

    # ── 6b. Write the paying-schools roster ──────────────────────────────────
    roster_data = {
        "generated_at":    now.isoformat(timespec="seconds"),
        "generated_label": timestamp,
        "filter": {
            "pipeline":        P_CAM,
            "stages":          [f"{S_ONBOARDING} (Onboarding)", f"{S_ACTIVATED} (Activated)",
                                 f"{S_UPCOMING_RENEWAL} (Upcoming Renewal)", f"{S_RENEWED} (Renewed)"],
            "currency":        "zar",
            "exclude":         "B2C accounts (deal title or account name contains 'B2C'); "
                                "automation test deals (title starts with 'ZZZ')",
        },
        "count":   len(roster),
        "schools": roster,
    }
    with open(ROSTER_FILE, "w", encoding="utf-8") as f:
        json.dump(roster_data, f, indent=2, ensure_ascii=False)
    print(f"✅  paying_schools.json updated ({len(roster)} schools)")

    # ── 6c. Write the confirmed-renewals list ────────────────────────────────
    renewals_data = {
        "generated_at":    now.isoformat(timespec="seconds"),
        "generated_label": timestamp,
        "filter": {
            "pipeline":  P_CAM,
            "stage":     f"{S_RENEWED} (Renewed) only",
            "currency":  "zar",
            "exclude":   "B2C accounts (deal title or account name contains 'B2C')"
                         + (" -- automation test deals (title starts with 'ZZZ') currently"
                            " INCLUDED as placeholder data, see RENEWALS_INCLUDE_TEST_DEALS"
                            " in update_sales_dashboard.py"
                            if RENEWALS_INCLUDE_TEST_DEALS else
                            "; automation test deals (title starts with 'ZZZ')"),
            "note":      "Does NOT include 'Upcoming Renewal' (52) -- that stage means "
                         "the cycle is approaching, not that the renewal is confirmed.",
        },
        "test_data_included": RENEWALS_INCLUDE_TEST_DEALS,
        "count":            len(renewals),
        "total_value_zar":  sum(r["value_zar"] for r in renewals),
        "renewals":         renewals,
        "upcoming": {
            "filter": {
                "pipeline": P_CAM,
                "stage":    f"{S_UPCOMING_RENEWAL} (Upcoming Renewal) only",
                "currency": "zar",
                "note":     "Renewal cycle approaching but NOT yet confirmed -- "
                            "these schools are still active paying customers.",
            },
            "count":           len(upcoming_renewals),
            "total_value_zar": sum(r["value_zar"] for r in upcoming_renewals),
            "schools":         upcoming_renewals,
        },
    }
    with open(RENEWALS_FILE, "w", encoding="utf-8") as f:
        json.dump(renewals_data, f, indent=2, ensure_ascii=False)
    print(f"✅  renewals_data.json updated "
          f"({len(renewals)} confirmed, {len(upcoming_renewals)} due for renewal)")

    # ── 7. Git commit & push ─────────────────────────────────────────────────
    if no_push:
        print("\n🛑  --no-push set — skipping git commit + push.")
        print("    JSON files saved locally. Inspect them before pushing.")
    elif not pull_ok:
        print("\n🛑  Skipping push — the pull at the start of this run failed.")
        print("    JSON files were saved locally. Run 'git pull origin main' by hand,")
        print("    resolve anything it reports, then 'git push origin main'.")
    else:
        print("\n🚀  Publishing to GitHub…")
        try:
            subprocess.run(["git", "-C", str(SCRIPT_DIR), "add",
                            "pipeline_data.json", "paying_schools.json", "renewals_data.json",
                            "investor.html", "renewals.html"], check=True)
            subprocess.run(["git", "-C", str(SCRIPT_DIR), "commit", "-m",
                            f"Update sales dashboard — {timestamp}"], check=True)
            subprocess.run(["git", "-C", str(SCRIPT_DIR), "push", "origin", "main"], check=True)
            print("✅  Pushed! Live on GitHub Pages within ~60 seconds.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git error: {e}")
            print("    JSON files were saved locally — push manually when ready.")

    print("\n🎉  Done! Dashboard updated safely.")
    print("    Manual sections preserved: Product Demos, Schools list")
    print(f"    View live at: https://earacademy.github.io/Ear-Academy-Usage-Tracker/investor.html\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update Ear Academy sales dashboard from ActiveCampaign.")
    parser.add_argument("--no-push", action="store_true",
                        help="Write JSON files locally but skip the git commit + push step (testing).")
    args = parser.parse_args()
    main(no_push=args.no_push)
