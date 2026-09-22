#!/usr/bin/env python3
"""
Ear Academy — Motion A / Stale-Pipeline Cross-Check
=====================================================
Answers one question: of the deals currently stale (velocity_data.json,
Pipeline 4 New Lead, 90+ days), which has Rus already worked through Motion A
(the Supabase-tracked cold outreach motion), and which are untouched?

Earlier cross-checks were built from a one-time Motion A CSV export and a
point-in-time stale list, so the match silently went stale itself (e.g. a
deal that had moved to Long Term Interest kept showing as an unworked stale
lead). This script always pulls both sides live, so the comparison is only
ever as stale as the last run.

Every candidate name match is verified against a live ActiveCampaign lookup
before being reported — a fuzzy string match alone is not proof the two
records refer to the same school (e.g. "Waterfall College" and "Waterfall
Prep" score as a near-match but are different schools).

SAFE BY DESIGN:
  ✅  Read-only against Supabase and ActiveCampaign
  🚫  Never writes to velocity_data.json or any dashboard file — writes only
      motion_a_crosscheck.json

Run:
  cd ~/Desktop/ear-academy-analytics && python3 motion_a_crosscheck.py
"""

import json
import re
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from config import AC_API_KEY, AC_BASE_URL
except ImportError:
    print("❌  Could not find config.py. Make sure it is in the same folder as this script.")
    sys.exit(1)

try:
    from supabase_config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY
except ImportError:
    print("❌  Could not find supabase_config.py. Make sure it is in the same folder as this script.")
    sys.exit(1)

OUTPUT_FILE = SCRIPT_DIR / "motion_a_crosscheck.json"
VELOCITY_FILE = SCRIPT_DIR / "velocity_data.json"
TODAY = datetime.now(timezone.utc)

AC_HEADERS = {"Api-Token": AC_API_KEY, "Content-Type": "application/json"}
SB_HEADERS = {"apikey": SUPABASE_PUBLISHABLE_KEY, "Authorization": f"Bearer {SUPABASE_PUBLISHABLE_KEY}"}

# Motion A outreach funnel, least to most advanced — used to collapse the
# same school's repeated rows (motion_a_daily logs a new row per status
# change rather than updating one in place) down to its latest stage.
STATUS_RANK = {
    "t1-sent": 1, "t2-sent": 2, "t3-sent": 3,
    "reply-received": 4, "demo-held": 5, "negotiation": 6, "close": 7,
}

NAME_MATCH_THRESHOLD = 0.85


# ── ActiveCampaign ──────────────────────────────────────────────────────────

def ac_get(endpoint, params=None):
    url = f"{AC_BASE_URL.rstrip('/')}/api/3/{endpoint}"
    r = requests.get(url, headers=AC_HEADERS, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def ac_search_deals(title_fragment):
    """Substring search across deal titles. filters[title] only does exact
    match — filters[search] is the one that actually does substring search."""
    data = ac_get("deals", {"filters[search]": title_fragment, "limit": 20})
    return data.get("deals", [])


def ac_get_deal(deal_id):
    try:
        data = ac_get(f"deals/{deal_id}")
        return data.get("deal")
    except requests.HTTPError:
        return None


# ── Supabase (Motion A) ─────────────────────────────────────────────────────

def fetch_motion_a_rows():
    """Pull every row from the motion_a_daily table."""
    rows = []
    offset = 0
    while True:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/motion_a_daily",
            headers={**SB_HEADERS, "Range": f"{offset}-{offset + 99}"},
            params={"select": "id,school_name,status,ac_deal_id"},
            timeout=15,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < 100:
            break
    return rows


def dedupe_motion_a(rows):
    """Collapse to one row per school, keeping the most-advanced status."""
    best = {}
    for r in rows:
        name = (r.get("school_name") or "").strip()
        if not name:
            continue
        rank = STATUS_RANK.get(r.get("status"), 0)
        cur = best.get(name)
        if cur is None or rank > cur["_rank"]:
            best[name] = {**r, "_rank": rank}
        elif rank == cur["_rank"] and r.get("ac_deal_id") and not cur.get("ac_deal_id"):
            best[name] = {**r, "_rank": rank}
    return list(best.values())


# ── Matching ─────────────────────────────────────────────────────────────────

def normalise(name):
    n = (name or "").lower()
    n = re.sub(r"[’'\-\(\)\.,]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def fuzzy_score(a, b):
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def load_stale_deals():
    with open(VELOCITY_FILE) as f:
        data = json.load(f)
    return data["pipeline4"]["stale_deals"]


# ── Verification ─────────────────────────────────────────────────────────────

def verify_match(stale_deal, motion_a_row):
    """Confirm a candidate name match against live AC. Returns a dict with
    a 'verified' bool and the evidence used, rather than trusting the fuzzy
    score or Motion A's own (possibly stale) status field."""
    deal_id = motion_a_row.get("ac_deal_id")
    live_deal = ac_get_deal(deal_id) if deal_id else None

    if live_deal:
        # Motion A points at a specific deal — confirm it's the SAME deal
        # the stale list flagged, not just a plausibly-similar one.
        same_deal = str(live_deal.get("id")) == str(stale_deal["id"])
        return {
            "verified": same_deal,
            "method": "ac_deal_id lookup",
            "live_title": live_deal.get("title"),
            "live_group": live_deal.get("group"),
            "live_stage": live_deal.get("stage"),
            "note": "Motion A's ac_deal_id points at a different deal than the stale-list entry" if not same_deal else None,
        }

    # No deal id on the Motion A side — confirm independently via title search
    hits = ac_search_deals(motion_a_row["school_name"])
    hit = next((h for h in hits if str(h.get("id")) == str(stale_deal["id"])), None)
    return {
        "verified": hit is not None,
        "method": "title search",
        "live_title": hit.get("title") if hit else None,
        "live_group": hit.get("group") if hit else None,
        "live_stage": hit.get("stage") if hit else None,
        "note": None if hit else "No live AC deal matching this title/id combination was found",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Motion A / Stale-Pipeline Cross-Check")
    print("=" * 60)

    print("\n📥  Pulling Motion A rows from Supabase…")
    raw_rows = fetch_motion_a_rows()
    print(f"    → {len(raw_rows)} rows")

    motion_a = dedupe_motion_a(raw_rows)
    print(f"    → {len(motion_a)} unique schools after dedupe "
          f"({len(raw_rows) - len(motion_a)} repeat-status rows collapsed)")

    print("\n📥  Loading fresh stale-deals list from velocity_data.json…")
    stale_deals = load_stale_deals()
    print(f"    → {len(stale_deals)} stale deals (90d+ in New Lead)")

    print("\n🔍  Fuzzy-matching stale deals against Motion A school names…")
    candidates = []
    for deal in stale_deals:
        for row in motion_a:
            score = fuzzy_score(deal["title"], row["school_name"])
            if score >= NAME_MATCH_THRESHOLD:
                candidates.append((deal, row, score))
    print(f"    → {len(candidates)} candidate match(es) found")

    print("\n✅  Verifying each candidate against live ActiveCampaign…")
    results = []
    for deal, row, score in candidates:
        v = verify_match(deal, row)
        status = "VERIFIED" if v["verified"] else "REJECTED"
        print(f"    [{status}] '{deal['title']}' ~ '{row['school_name']}' "
              f"(score {score:.2f}, {v['method']})")
        results.append({
            "stale_deal": {"id": deal["id"], "title": deal["title"], "days_in_stage": deal["days_in_stage"]},
            "motion_a": {"school_name": row["school_name"], "status": row["status"], "ac_deal_id": row.get("ac_deal_id")},
            "match_score": round(score, 3),
            **v,
        })

    verified = [r for r in results if r["verified"]]
    rejected = [r for r in results if not r["verified"]]
    matched_deal_ids = {r["stale_deal"]["id"] for r in verified}
    untouched = [d for d in stale_deals if d["id"] not in matched_deal_ids]

    print(f"\n📊  Summary:")
    print(f"    Verified matches (Motion A has worked this stale deal): {len(verified)}")
    print(f"    Rejected candidates (fuzzy match, not the same deal):   {len(rejected)}")
    print(f"    Stale deals with NO Motion A contact found:             {len(untouched)}")

    output = {
        "generated_at": TODAY.isoformat(),
        "stale_deal_count": len(stale_deals),
        "motion_a_school_count": len(motion_a),
        "verified_matches": verified,
        "rejected_candidates": rejected,
        "untouched_stale_deals": [{"id": d["id"], "title": d["title"], "days_in_stage": d["days_in_stage"]} for d in untouched],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾  Saved to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
