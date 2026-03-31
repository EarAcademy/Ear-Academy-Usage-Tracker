#!/usr/bin/env python3
"""
Generates pipeline_data.json for the investor dashboard.
Pulls live deal/pipeline data from ActiveCampaign API.

Usage:
  AC_API_KEY=your_key python3 generate_pipeline_data.py

Or set AC_API_KEY in your environment before running.
"""

import os, json, requests
from datetime import datetime, timedelta, timezone

AC_API_URL = "https://the-ear.api-us1.com/api/3"
AC_API_KEY = os.environ.get("AC_API_KEY", "")
OUTPUT_FILE = "pipeline_data.json"

def ac_get(endpoint, params=None):
    headers = {"Api-Token": AC_API_KEY}
    url = f"{AC_API_URL}/{endpoint}"
    resp = requests.get(url, headers=headers, params=params or {})
    resp.raise_for_status()
    return resp.json()

CHURNED_DEALS = [
    'academie orfeus',
    'the lighthouse learning hub',
    'king david',
]

def is_churned(deal_title):
    """Return True if deal title matches a known churned school."""
    title_lower = deal_title.lower()
    return any(churned in title_lower for churned in CHURNED_DEALS)

def is_agreed_stage(name_lower):
    """Return True if a stage name maps to the Agreed bucket."""
    return (
        "agreed" in name_lower or "won" in name_lower or "closed" in name_lower
        or "onboarding" in name_lower or "activated" in name_lower
        or "upcoming renewal" in name_lower or "low activity" in name_lower
        or "churning" in name_lower
    )

def main():
    if not AC_API_KEY:
        print("⚠️  AC_API_KEY not set. Skipping pipeline data generation.")
        return

    now_utc = datetime.now(timezone.utc)
    seven_days_ago = now_utc - timedelta(days=7)
    seven_days_ago_str = (now_utc - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00")

    print("Fetching pipeline stages from ActiveCampaign...")

    # ── Deal stages ──────────────────────────────────────────────────────────
    stages_data = ac_get("dealStages", {"limit": 100})
    stage_map = {s["id"]: s["title"] for s in stages_data.get("dealStages", [])}

    # ── All deals (paginated) ─────────────────────────────────────────────────
    all_deals = []
    offset = 0
    while True:
        resp = ac_get("deals", {"limit": 100, "offset": offset})
        deals = resp.get("deals", [])
        if not deals:
            break
        all_deals.extend(deals)
        offset += len(deals)
        if len(deals) < 100:
            break

    print(f"  Found {len(all_deals)} total deals")

    # ── Count by stage + weekly activity ─────────────────────────────────────
    stage_counts = {}
    stage_deals  = {}

    standard_stages = {
        "New Lead":    0,
        "Demo/Pilot":  0,
        "Negotiation": 0,
        "Agreed":      0,
    }

    weekly_advances      = 0   # deals moved into Demo/Pilot or Negotiation in last 7 days
    weekly_new_customers = 0   # deals moved into an Agreed-type stage in last 7 days

    for deal in all_deals:
        stage_id   = deal.get("stage", "")
        stage_name = stage_map.get(stage_id, f"Stage {stage_id}")
        name_lower = stage_name.lower()

        # Raw stage counts (all_stages)
        stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
        if stage_name not in stage_deals:
            stage_deals[stage_name] = []

        mdate = deal.get("mdate", "")
        days_in_stage = None
        is_recent = False
        if mdate:
            try:
                mdate_dt = datetime.fromisoformat(mdate.replace("Z", "+00:00"))
                days_in_stage = (now_utc - mdate_dt).days
                is_recent = mdate_dt >= seven_days_ago
            except Exception:
                pass

        stage_deals[stage_name].append({
            "title":         deal.get("title", ""),
            "days_in_stage": days_in_stage,
            "value":         deal.get("value", 0),
        })

        # Map to 4 standard stages
        if "new" in name_lower or "lead" in name_lower:
            standard_stages["New Lead"] += 1
        elif "demo" in name_lower or "pilot" in name_lower:
            standard_stages["Demo/Pilot"] += 1
            if is_recent:
                weekly_advances += 1
        elif "negotiat" in name_lower:
            standard_stages["Negotiation"] += 1
            if is_recent:
                weekly_advances += 1
        elif is_agreed_stage(name_lower):
            if not is_churned(deal.get("title", "")):
                standard_stages["Agreed"] += 1
                if is_recent:
                    weekly_new_customers += 1

    # ── Weekly: new leads (contacts created in last 7 days) ──────────────────
    print("Fetching weekly new leads...")
    weekly_new_leads = "—"
    try:
        resp = ac_get("contacts", {
            "limit": 1,
            "filters[created_after]": seven_days_ago_str,
        })
        total = resp.get("meta", {}).get("total", None)
        if total is not None:
            weekly_new_leads = str(total)
    except Exception as e:
        print(f"  Weekly new leads error: {e}")

    # ── Weekly: emails sent (campaigns delivered in last 7 days) ─────────────
    print("Fetching weekly email sends...")
    weekly_emails = "—"
    try:
        campaigns_resp = ac_get("campaigns", {"limit": 50, "orders[sdate]": "desc"})
        campaigns = campaigns_resp.get("campaigns", [])
        cutoff_naive = datetime.now() - timedelta(days=7)
        total_sent = 0
        for c in campaigns:
            sdate = c.get("sdate", "")
            if not sdate:
                continue
            try:
                sdate_dt = datetime.fromisoformat(sdate.replace("Z", "+00:00")).replace(tzinfo=None)
                if sdate_dt >= cutoff_naive:
                    total_sent += int(c.get("send_amt", 0) or 0)
            except Exception:
                pass
        if total_sent:
            weekly_emails = f"{total_sent:,}"
    except Exception as e:
        print(f"  Weekly emails error: {e}")

    # ── Total contacts ────────────────────────────────────────────────────────
    print("Fetching total contacts...")
    total_contacts = 0
    try:
        resp = ac_get("contacts", {"limit": 1})
        total_contacts = int(resp.get("meta", {}).get("total", 0))
    except Exception as e:
        print(f"  Total contacts error: {e}")

    # ── Email campaign stats (last 10 campaigns, all-time totals) ─────────────
    print("Fetching email campaign stats...")
    email_data = {"sent": "—", "replied": "—", "rate": "—", "new_contacts": "—"}
    try:
        this_month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%S")
        contacts_resp = ac_get("contacts", {"limit": 1, "filters[created_after]": this_month_start})
        month_contacts = contacts_resp.get("meta", {}).get("total", "—")

        campaigns_resp = ac_get("campaigns", {"limit": 20, "orders[sdate]": "desc"})
        campaigns = campaigns_resp.get("campaigns", [])
        total_sent  = sum(int(c.get("send_amt", 0) or 0) for c in campaigns[:10])
        total_opens = sum(int(c.get("opens",    0) or 0) for c in campaigns[:10])

        email_data = {
            "sent":         f"{total_sent:,}"  if total_sent  else "—",
            "replied":      f"{total_opens:,}" if total_opens else "—",
            "rate":         f"{round(total_opens/total_sent*100, 1)}%" if total_sent and total_opens else "—",
            "new_contacts": str(month_contacts) if month_contacts != "—" else "—",
        }
    except Exception as e:
        print(f"  Email stats error: {e}")

    # ── Monthly new leads (Jan / Feb / Mar 2026) ──────────────────────────────
    print("Fetching monthly lead counts...")
    monthly = {}
    month_ranges = [
        ("jan", "2026-01-01T00:00:00", "2026-02-01T00:00:00"),
        ("feb", "2026-02-01T00:00:00", "2026-03-01T00:00:00"),
        ("mar", "2026-03-01T00:00:00", "2026-04-01T00:00:00"),
    ]
    for label, start, end in month_ranges:
        try:
            resp = ac_get("contacts", {
                "limit": 1,
                "filters[created_after]":  start,
                "filters[created_before]": end,
            })
            count = resp.get("meta", {}).get("total", "—")
            monthly[label] = {
                "new_leads": str(count) if count != "—" else "—",
                "new_deals": "—",
            }
        except Exception as e:
            print(f"  Monthly {label} error: {e}")
            monthly[label] = {"new_leads": "—", "new_deals": "—"}

    # ── Assemble output ───────────────────────────────────────────────────────
    output = {
        "stages": standard_stages,
        "all_stages": stage_counts,
        "weekly": {
            "new_leads":    weekly_new_leads,
            "emails_sent":  weekly_emails,
            "advances":     str(weekly_advances) if weekly_advances else "—",
            "new_customers": str(weekly_new_customers) if weekly_new_customers else "—",
        },
        "monthly": monthly,
        "email": email_data,
        "last_updated":   datetime.now().strftime("%d %b %Y at %H:%M"),
        "total_deals":    len(all_deals),
        "total_contacts": total_contacts,
        "notes": {
            "New Lead":      "Active deals in Sales Qualification pipeline, New Lead stage (across ZAR/GBP/USD)",
            "Demo/Pilot":    "Active deals in Sales Conversion pipeline, Demo/Pilot stage",
            "Negotiation":   "Active deals in Sales Conversion pipeline, Negotiation stage",
            "Agreed":        "Paying schools confirmed by Rus (Customer Account Management Won: Onboarding + Activated)",
            "advances":      "Deals modified into Demo/Pilot or Negotiation in last 7 days",
            "new_customers": "Non-churned deals modified into an Agreed-type stage in last 7 days",
            "email_sent":    "Sum of send_amt across last 10 campaigns",
            "email_replied": "Email opens across last 10 campaigns",
            "new_contacts":  "New contacts added this calendar month",
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Pipeline data saved to {OUTPUT_FILE}")
    print(f"   Stages:         {standard_stages}")
    print(f"   Weekly advances:     {weekly_advances}")
    print(f"   Weekly new customers:{weekly_new_customers}")
    print(f"   Weekly new leads:    {weekly_new_leads}")
    print(f"   Weekly emails sent:  {weekly_emails}")
    print(f"   Total deals:    {len(all_deals)}")
    print(f"   Total contacts: {total_contacts}")

if __name__ == "__main__":
    main()
