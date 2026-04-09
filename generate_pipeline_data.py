#!/usr/bin/env python3
"""
Ear Academy — Sales Dashboard Auto-Update Script
=================================================
Single command to update and publish the live dashboard:

    python3 generate_pipeline_data.py

What it does:
    1. Pulls all ZAR-only deals from ActiveCampaign (Pipelines 4, 5, 6)
    2. Pulls SA broadcast campaign stats
    3. Calculates all dashboard metrics
    4. Rebuilds investor.html with live data
    5. Pushes to GitHub -> live within 60 seconds

Requires:
    - config.py in the same folder (with AC_API_KEY and AC_BASE_URL)
    - pip3 install requests
    - Git configured with SSH key
"""

import json
import os
import sys
import subprocess
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# Config
try:
    import config
    AC_API_KEY  = config.AC_API_KEY
    AC_BASE_URL = config.AC_BASE_URL
except ImportError:
    print("ERROR: config.py not found.")
    print("Create config.py with:")
    print('AC_API_KEY  = "your-key-here"')
    print('AC_BASE_URL = "https://the-ear.activehosted.com"')
    sys.exit(1)

if not AC_API_KEY or AC_API_KEY == "your-key-here":
    print("ERROR: AC_API_KEY not set in config.py")
    sys.exit(1)

# Constants
SCRIPT_DIR  = Path(__file__).parent
DASHBOARD   = SCRIPT_DIR / "investor.html"
TODAY       = datetime.now().strftime("%d %B %Y")
TODAY_ISO   = datetime.now().strftime("%Y-%m-%d")
THIS_MONTH  = datetime.now().strftime("%Y-%m")
LAST_MONTH  = (datetime.now().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")

P_QUAL = "4"
P_CONV = "5"
P_CAM  = "6"

S_NEW_LEAD    = "36"
S_DEMO_PILOT  = "43"
S_NEGOTIATION = "46"
S_ONBOARDING  = "50"
S_ACTIVATED   = "51"

OPEN = "0"
WON  = "1"
LOST = "2"


def ac_get(endpoint, params=None):
    headers = {"Api-Token": AC_API_KEY}
    base    = f"{AC_BASE_URL}/api/3/{endpoint}"
    params  = params or {}
    params["limit"] = 100
    results = []
    offset  = 0
    while True:
        params["offset"] = offset
        try:
            r = requests.get(base, headers=headers, params=params, timeout=30)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  API error on {endpoint}: {e}")
            break
        data  = r.json()
        key   = [k for k in data if isinstance(data[k], list)][0]
        batch = data[key]
        results.extend(batch)
        total  = int(data.get("meta", {}).get("total", len(batch)))
        offset += len(batch)
        if offset >= total or not batch:
            break
    return results


def fetch_zar_deals(status=None, group=None):
    params = {}
    if status is not None:
        params["filters[status]"] = status
    if group is not None:
        params["filters[group]"] = group
    deals = ac_get("deals", params)
    return [d for d in deals if d.get("currency", "").lower() == "zar"]


def fetch_sa_campaigns():
    campaigns = ac_get("campaigns", {"filters[status]": "5"})
    sa = []
    for c in campaigns:
        name = c.get("name", "")
        sent = int(c.get("send_amt", 0))
        if sent > 5 and "UK" not in name and "Test" not in name and "JK" not in name:
            if any(kw in name for kw in ["SA", "South", "Webinar", "Mailer", "Reengagement", "Schools", "Musical"]):
                opens = int(c.get("uniqueopens", 0))
                sa.append({
                    "name":  name,
                    "date":  (c.get("ldate") or c.get("sdate") or "")[:10],
                    "sent":  sent,
                    "opens": opens,
                    "rate":  round(opens / sent * 100) if sent > 0 else 0,
                })
    sa.sort(key=lambda x: x["date"], reverse=True)
    return sa[:5]


def days_since(date_str):
    try:
        dt = datetime.fromisoformat(date_str[:19].replace("T", " "))
        return max(0, (datetime.now() - dt).days)
    except:
        return 0


def month_of(date_str):
    return (date_str or "")[:7]


def fmt_zar(cents):
    r = int(cents) // 100
    if r >= 1_000_000:
        return f"R{r/1_000_000:.1f}M"
    if r >= 1_000:
        return f"R{r/1_000:,.0f}k"
    return f"R{r:,}"


def last_n_months(n=6):
    months = []
    for i in range(n - 1, -1, -1):
        dt = (datetime.now().replace(day=1) - timedelta(days=i * 28)).replace(day=1)
        months.append(dt.strftime("%Y-%m"))
    return months


def month_label(ym):
    try:
        return datetime.strptime(ym, "%Y-%m").strftime("%b %y")
    except:
        return ym


def calculate(p4_open, p5_open, cam_won, cam_open, lost_zar, campaigns):
    m = {}
    months = last_n_months(6)
    m["month_labels"] = [month_label(mo) for mo in months]

    m["p4_count"]      = len(p4_open)
    m["p4_value"]      = fmt_zar(sum(int(d.get("value", 0)) for d in p4_open))
    m["p4_this_month"] = len([d for d in p4_open if month_of(d.get("cdate","")) == THIS_MONTH])
    m["p4_last_month"] = len([d for d in p4_open if month_of(d.get("cdate","")) == LAST_MONTH])

    leads_by_month = {mo: 0 for mo in months}
    for d in p4_open:
        mo = month_of(d.get("cdate", ""))
        if mo in leads_by_month:
            leads_by_month[mo] += 1
    m["leads_by_month"] = list(leads_by_month.values())

    p5_demo = [d for d in p5_open if d.get("stage") == S_DEMO_PILOT]
    p5_neg  = [d for d in p5_open if d.get("stage") == S_NEGOTIATION]
    m["p5_demo_count"] = len(p5_demo)
    m["p5_neg_count"]  = len(p5_neg)
    m["p5_count"]      = len(p5_open)
    m["p5_demo_val"]   = fmt_zar(sum(int(d.get("value", 0)) for d in p5_demo))
    m["p5_neg_val"]    = fmt_zar(sum(int(d.get("value", 0)) for d in p5_neg))

    all_customers = cam_won + cam_open
    m["customers_total"]      = len(all_customers)
    m["customers_onboarding"] = len([d for d in all_customers if d.get("stage") == S_ONBOARDING])
    m["customers_activated"]  = len([d for d in all_customers if d.get("stage") == S_ACTIVATED])
    arr_raw                   = sum(int(d.get("value", 0)) for d in all_customers)
    m["arr_str"]              = fmt_zar(arr_raw)
    m["arr_raw"]              = arr_raw

    t12 = [d for d in all_customers if int(d.get("value",0)) >= 1_000_000]
    t8  = [d for d in all_customers if 600_000 <= int(d.get("value",0)) < 1_000_000]
    tlo = [d for d in all_customers if int(d.get("value",0)) < 600_000]
    tot = max(m["customers_total"], 1)
    m["tier_12k"] = {"count": len(t12), "val": fmt_zar(sum(int(d.get("value",0)) for d in t12)), "pct": round(len(t12)/tot*100)}
    m["tier_8k"]  = {"count": len(t8),  "val": fmt_zar(sum(int(d.get("value",0)) for d in t8)),  "pct": round(len(t8) /tot*100)}
    m["tier_lo"]  = {"count": len(tlo), "val": fmt_zar(sum(int(d.get("value",0)) for d in tlo)), "pct": round(len(tlo)/tot*100)}

    won_by_month = {mo: 0 for mo in months}
    for d in all_customers:
        mo = month_of(d.get("edate","") or d.get("cdate",""))
        if mo in won_by_month:
            won_by_month[mo] += 1
    m["won_by_month"] = list(won_by_month.values())

    m["lost_conv"]  = len([d for d in lost_zar if d.get("group") == P_CONV])
    m["lost_qual"]  = len([d for d in lost_zar if d.get("group") == P_QUAL])
    m["lost_total"] = len(lost_zar)

    closed = m["customers_total"] + m["lost_total"]
    m["win_rate"]         = round(m["customers_total"] / max(closed, 1) * 100)
    m["qual_to_conv_pct"] = round(m["p5_count"] / max(m["p4_count"] + m["p5_count"], 1) * 100)
    m["demo_to_neg_pct"]  = round(m["p5_neg_count"] / max(m["p5_count"], 1) * 100)

    open_val = sum(int(d.get("value",0)) for d in p4_open + p5_open)
    m["pipeline_str"]      = fmt_zar(open_val)
    m["pipeline_coverage"] = round(open_val / max(arr_raw, 1), 1)

    delta = m["p4_this_month"] - m["p4_last_month"]
    m["leads_delta"]       = f"+{delta}" if delta >= 0 else str(delta)
    m["leads_delta_class"] = "up" if delta > 0 else "flat" if delta == 0 else "down"

    stage_names = {S_NEW_LEAD: "New Lead", S_DEMO_PILOT: "Demo / Pilot", S_NEGOTIATION: "Negotiation"}
    stuck = []
    for d in p4_open + p5_open:
        days = days_since(d.get("cdate",""))
        if days > 30:
            stuck.append({
                "name":    d.get("title","Unknown")[:55],
                "stage":   stage_names.get(d.get("stage",""), "Unknown"),
                "days":    days,
                "value":   fmt_zar(int(d.get("value",0))),
                "urgency": "Critical" if days > 150 else "Review" if days > 90 else "Monitor",
                "cls":     "pill-red" if days > 150 else "pill-amber" if days > 90 else "pill-blue",
            })
    stuck.sort(key=lambda x: x["days"], reverse=True)
    m["stuck"]     = stuck[:20]
    m["campaigns"] = campaigns
    return m


def render_campaigns(camps):
    if not camps:
        return "<p style='color:var(--muted);font-size:12px;padding:12px 0;'>No SA campaigns found.</p>"
    html = ""
    for c in camps:
        html += f'<div class="campaign-row"><div style="flex:1"><div class="campaign-name">{c["name"]}</div><div class="campaign-meta">{c["date"]} &middot; {c["sent"]:,} recipients</div></div><div class="campaign-stat"><div class="val">{c["sent"]:,}</div><div class="lbl">sent</div></div><div><div style="font-size:11px;color:var(--muted);margin-bottom:4px;">{c["rate"]}% open rate</div><div class="open-rate-bar"><div class="open-rate-fill" style="width:{min(c["rate"],100)}%"></div></div></div></div>'
    return html


def render_stuck(stuck):
    if not stuck:
        return "<tr><td colspan='5' style='text-align:center;color:var(--muted);padding:20px;'>No deals stuck &gt;30 days &mdash; pipeline is flowing well</td></tr>"
    html = ""
    for d in stuck:
        sc = "pill-blue" if d["stage"] == "Negotiation" else "pill-amber"
        html += f'<tr><td style="font-weight:500;">{d["name"]}</td><td><span class="pill {sc}">{d["stage"]}</span></td><td style="font-weight:600;">{d["days"]} days</td><td>{d["value"]}</td><td><span class="pill {d["cls"]}">{d["urgency"]}</span></td></tr>'
    return html


def build_html(m):
    lm = json.dumps(m["leads_by_month"])
    wm = json.dumps(m["won_by_month"])
    lb = json.dumps(m["month_labels"])

    css = """
:root{--lapis:#1d70b8;--sky:#38aae1;--green:#00a19a;--gold:#c5b08c;--shale:#d8d3cb;--white:#fafaf8;--ink:#1a1a1a;--muted:#6b6b6b;--surface:#f4f2ef;--border:#e4e0da;--warn:#d94f3b;--warn-bg:#fdf1ef;}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'DM Sans',sans-serif;background:var(--white);color:var(--ink);font-size:14px;line-height:1.5;}
header{background:var(--ink);padding:0 40px;display:flex;align-items:center;justify-content:space-between;height:64px;position:sticky;top:0;z-index:100;}
.logo{font-family:'DM Serif Display',serif;font-size:20px;color:var(--white);letter-spacing:-0.3px;}
.logo span{color:var(--sky);}
.badge{background:var(--green);color:white;font-size:10px;font-weight:600;padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:0.8px;}
.updated{font-size:11px;color:rgba(255,255,255,0.45);letter-spacing:0.5px;text-transform:uppercase;}
main{max-width:1280px;margin:0 auto;padding:40px 40px 80px;}
.page-title{font-family:'DM Serif Display',serif;font-size:13px;font-style:italic;color:var(--muted);margin-bottom:32px;display:flex;align-items:center;gap:12px;}
.page-title::after{content:'';flex:1;height:1px;background:var(--border);}
.hero-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:2px;margin-bottom:40px;background:var(--border);border:1px solid var(--border);border-radius:12px;overflow:hidden;}
.hero-card{background:var(--white);padding:28px 28px 24px;position:relative;}
.hero-card:first-child{border-radius:11px 0 0 11px;}.hero-card:last-child{border-radius:0 11px 11px 0;}
.hero-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:12px;}
.hero-value{font-family:'DM Serif Display',serif;font-size:42px;line-height:1;color:var(--ink);margin-bottom:8px;}
.hero-value.hl{color:var(--lapis);}.hero-value.gr{color:var(--green);}
.hero-sub{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:6px;}
.delta{font-size:11px;font-weight:600;padding:2px 7px;border-radius:4px;}
.delta.up{background:#e8f6f5;color:var(--green);}.delta.down{background:var(--warn-bg);color:var(--warn);}.delta.flat{background:#f0f0f0;color:var(--muted);}
.hero-accent{position:absolute;top:0;left:0;width:3px;height:100%;}
.section{margin-bottom:40px;}
.section-header{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:20px;padding-bottom:12px;border-bottom:1px solid var(--border);}
.section-title{font-family:'DM Serif Display',serif;font-size:20px;}
.section-note{font-size:11px;color:var(--muted);font-style:italic;}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:24px;}
.card{background:var(--white);border:1px solid var(--border);border-radius:10px;padding:24px;}
.card-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:var(--muted);margin-bottom:20px;}
.funnel{display:flex;gap:2px;margin-bottom:28px;}
.funnel-stage{flex:1;background:var(--surface);border-radius:6px;padding:16px 14px;position:relative;}
.funnel-stage::after{content:'rarr';position:absolute;right:-12px;top:50%;transform:translateY(-50%);color:var(--shale);font-size:14px;z-index:2;}
.funnel-stage:last-child::after{display:none;}
.funnel-stage.active{background:#edf5fb;}.funnel-stage.won{background:#e8f6f5;}
.funnel-stage-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:var(--muted);margin-bottom:6px;}
.funnel-stage-count{font-family:'DM Serif Display',serif;font-size:28px;color:var(--ink);line-height:1;margin-bottom:2px;}
.funnel-stage-val{font-size:11px;color:var(--muted);}
.funnel-stage-sub{margin-top:8px;font-size:10px;color:var(--lapis);font-weight:600;}
.stats-row{display:flex;}
.stat-box{flex:1;padding:20px;border-right:1px solid var(--border);}
.stat-box:last-child{border-right:none;}
.stat-label{font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--muted);font-weight:600;margin-bottom:8px;}
.stat-val{font-family:'DM Serif Display',serif;font-size:26px;color:var(--ink);line-height:1;margin-bottom:4px;}
.stat-sub{font-size:11px;color:var(--muted);}
.bar-chart{display:flex;align-items:flex-end;gap:8px;height:120px;padding-bottom:24px;border-bottom:1px solid var(--border);}
.bar-wrap{flex:1;display:flex;flex-direction:column;align-items:center;height:100%;justify-content:flex-end;}
.bar{width:100%;border-radius:4px 4px 0 0;min-height:2px;position:relative;}
.bar-val{font-size:10px;font-weight:600;position:absolute;top:-18px;left:50%;transform:translateX(-50%);white-space:nowrap;}
.bar-label{font-size:9px;color:var(--muted);position:absolute;bottom:-22px;left:50%;transform:translateX(-50%);white-space:nowrap;}
.flag-table{width:100%;border-collapse:collapse;}
.flag-table th{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.7px;color:var(--muted);text-align:left;padding:8px 12px;background:var(--surface);border-bottom:1px solid var(--border);}
.flag-table td{padding:10px 12px;font-size:12px;border-bottom:1px solid var(--border);vertical-align:middle;}
.flag-table tr:last-child td{border-bottom:none;}.flag-table tr:hover td{background:var(--surface);}
.pill{display:inline-block;font-size:10px;font-weight:600;padding:2px 8px;border-radius:20px;}
.pill-red{background:var(--warn-bg);color:var(--warn);}.pill-amber{background:#fff8e6;color:#b07d00;}.pill-green{background:#e8f6f5;color:var(--green);}.pill-blue{background:#edf5fb;color:var(--lapis);}
.campaign-row{display:flex;align-items:center;padding:12px 0;border-bottom:1px solid var(--border);gap:16px;}
.campaign-row:last-child{border-bottom:none;}
.campaign-name{flex:1;font-size:13px;font-weight:500;}
.campaign-meta{font-size:11px;color:var(--muted);margin-top:2px;}
.campaign-stat{text-align:right;min-width:60px;}
.campaign-stat .val{font-size:13px;font-weight:600;}.campaign-stat .lbl{font-size:10px;color:var(--muted);}
.open-rate-bar{width:80px;height:4px;background:var(--border);border-radius:2px;overflow:hidden;}
.open-rate-fill{height:100%;border-radius:2px;background:var(--sky);}
.arr-tier{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border);}
.arr-tier:last-child{border-bottom:none;}
.arr-tier-label{flex:1;font-size:12px;}.arr-tier-count{font-size:11px;color:var(--muted);min-width:60px;text-align:right;}.arr-tier-val{font-size:12px;font-weight:600;min-width:80px;text-align:right;}
.arr-bar{width:100px;height:4px;background:var(--border);border-radius:2px;overflow:hidden;}
.arr-bar-fill{height:100%;border-radius:2px;}
footer{border-top:1px solid var(--border);padding:20px 40px;display:flex;justify-content:space-between;font-size:11px;color:var(--muted);max-width:1280px;margin:0 auto;}
@media(max-width:900px){main{padding:24px 20px 60px;}.hero-grid{grid-template-columns:1fr 1fr;}.two-col{grid-template-columns:1fr;}.funnel{flex-direction:column;}.funnel-stage::after{display:none;}}
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>the eAr Academy - Sales Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
<header>
  <div class="logo">the e<span>A</span>r&#x2122; Academy</div>
  <div style="display:flex;align-items:center;gap:24px;">
    <span class="updated">SA Sales Dashboard &middot; {TODAY}</span>
    <span class="badge">Live</span>
  </div>
</header>
<main>
<div class="page-title">South Africa &mdash; ZAR Pipeline &amp; Sales Activity</div>
<div class="hero-grid">
  <div class="hero-card"><div class="hero-accent" style="background:var(--green)"></div><div class="hero-label">Active Customers</div><div class="hero-value gr">{m["customers_total"]}</div><div class="hero-sub">{m["customers_activated"]} activated &middot; {m["customers_onboarding"]} onboarding</div></div>
  <div class="hero-card"><div class="hero-accent" style="background:var(--lapis)"></div><div class="hero-label">Annual Recurring Revenue</div><div class="hero-value hl">{m["arr_str"]}</div><div class="hero-sub">ZAR &middot; SA schools only</div></div>
  <div class="hero-card"><div class="hero-accent" style="background:var(--sky)"></div><div class="hero-label">Open Pipeline Value</div><div class="hero-value">{m["pipeline_str"]}</div><div class="hero-sub">{m["pipeline_coverage"]}&times; coverage vs ARR</div></div>
  <div class="hero-card"><div class="hero-accent" style="background:var(--gold)"></div><div class="hero-label">New Leads This Month</div><div class="hero-value">{m["p4_this_month"]}</div><div class="hero-sub"><span class="delta {m["leads_delta_class"]}">{m["leads_delta"]}</span><span>vs last month ({m["p4_last_month"]})</span></div></div>
</div>
<div class="section">
  <div class="section-header"><h2 class="section-title">Sales Pipeline</h2><span class="section-note">ZAR only &middot; Open deals &middot; Live from ActiveCampaign</span></div>
  <div class="funnel">
    <div class="funnel-stage"><div class="funnel-stage-label">New Lead</div><div class="funnel-stage-count">{m["p4_count"]}</div><div class="funnel-stage-val">{m["p4_value"]} potential</div><div class="funnel-stage-sub">Sales Qualification</div></div>
    <div class="funnel-stage active"><div class="funnel-stage-label">Demo / Pilot</div><div class="funnel-stage-count">{m["p5_demo_count"]}</div><div class="funnel-stage-val">{m["p5_demo_val"]} potential</div><div class="funnel-stage-sub">Conversion Step 1</div></div>
    <div class="funnel-stage active"><div class="funnel-stage-label">Negotiation</div><div class="funnel-stage-count">{m["p5_neg_count"]}</div><div class="funnel-stage-val">{m["p5_neg_val"]} potential</div><div class="funnel-stage-sub">Conversion Step 2</div></div>
    <div class="funnel-stage won"><div class="funnel-stage-label">Won &middot; Active</div><div class="funnel-stage-count">{m["customers_total"]}</div><div class="funnel-stage-val">{m["arr_str"]} ARR</div><div class="funnel-stage-sub">Customer Account Mgmt</div></div>
  </div>
  <div class="card" style="padding:0;overflow:hidden;"><div class="stats-row">
    <div class="stat-box"><div class="stat-label">Qual &rarr; Conversion</div><div class="stat-val">{m["qual_to_conv_pct"]}%</div><div class="stat-sub">{m["p5_count"]} in conversion / {m["p4_count"]} in qual</div></div>
    <div class="stat-box"><div class="stat-label">Demo &rarr; Negotiation</div><div class="stat-val">{m["demo_to_neg_pct"]}%</div><div class="stat-sub">{m["p5_neg_count"]} of {m["p5_count"]} progressed</div></div>
    <div class="stat-box"><div class="stat-label">Win Rate (Closed)</div><div class="stat-val">{m["win_rate"]}%</div><div class="stat-sub">{m["customers_total"]} won vs {m["lost_total"]} lost</div></div>
    <div class="stat-box"><div class="stat-label">Pipeline Coverage</div><div class="stat-val">{m["pipeline_coverage"]}&times;</div><div class="stat-sub">{m["pipeline_str"]} open vs {m["arr_str"]} ARR</div></div>
  </div></div>
</div>
<div class="section">
  <div class="section-header"><h2 class="section-title">Sales Activity</h2><span class="section-note">New ZAR leads entering qualification &middot; Last 6 months</span></div>
  <div class="two-col">
    <div class="card"><div class="card-title">New ZAR Leads per Month</div><div class="bar-chart" id="leadsChart"></div></div>
    <div class="card"><div class="card-title">SA Broadcast Campaigns</div>{render_campaigns(m["campaigns"])}</div>
  </div>
</div>
<div class="section">
  <div class="section-header"><h2 class="section-title">Customer Account Management</h2><span class="section-note">All Won ZAR deals &middot; Pipeline 6</span></div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">Customers by Stage</div>
      <div style="display:flex;gap:16px;margin-bottom:24px;">
        <div style="flex:1;background:var(--surface);border-radius:8px;padding:20px;text-align:center;"><div style="font-family:'DM Serif Display',serif;font-size:36px;color:var(--sky);line-height:1;margin-bottom:4px;">{m["customers_onboarding"]}</div><div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--muted);font-weight:600;">Onboarding</div></div>
        <div style="flex:1;background:#e8f6f5;border-radius:8px;padding:20px;text-align:center;"><div style="font-family:'DM Serif Display',serif;font-size:36px;color:var(--green);line-height:1;margin-bottom:4px;">{m["customers_activated"]}</div><div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--muted);font-weight:600;">Activated</div></div>
      </div>
      <div class="card-title">ARR by Tier</div>
      <div class="arr-tier"><div class="arr-tier-label">R12,000 / year</div><div class="arr-bar"><div class="arr-bar-fill" style="width:{m["tier_12k"]["pct"]}%;background:var(--lapis)"></div></div><div class="arr-tier-count">{m["tier_12k"]["count"]} schools</div><div class="arr-tier-val">{m["tier_12k"]["val"]}</div></div>
      <div class="arr-tier"><div class="arr-tier-label">R8,050 / year</div><div class="arr-bar"><div class="arr-bar-fill" style="width:{m["tier_8k"]["pct"]}%;background:var(--sky)"></div></div><div class="arr-tier-count">{m["tier_8k"]["count"]} schools</div><div class="arr-tier-val">{m["tier_8k"]["val"]}</div></div>
      <div class="arr-tier"><div class="arr-tier-label">&lt; R8,050 / year</div><div class="arr-bar"><div class="arr-bar-fill" style="width:{m["tier_lo"]["pct"]}%;background:var(--shale)"></div></div><div class="arr-tier-count">{m["tier_lo"]["count"]} schools</div><div class="arr-tier-val">{m["tier_lo"]["val"]}</div></div>
      <div class="arr-tier" style="border-top:1px solid var(--border);margin-top:4px;"><div class="arr-tier-label" style="font-weight:600;">Total ARR</div><div class="arr-bar"></div><div class="arr-tier-count" style="font-weight:600;">{m["customers_total"]} schools</div><div class="arr-tier-val" style="color:var(--green);font-weight:700;">{m["arr_str"]}</div></div>
    </div>
    <div class="card">
      <div class="card-title">Customers Won by Month</div>
      <div class="bar-chart" id="wonChart"></div>
      <div style="margin-top:28px;"><div class="card-title">Lost Deals (ZAR)</div>
        <div style="display:flex;gap:12px;margin-bottom:16px;">
          <div style="flex:1;text-align:center;padding:14px;background:var(--warn-bg);border-radius:8px;"><div style="font-family:'DM Serif Display',serif;font-size:28px;color:var(--warn);">{m["lost_conv"]}</div><div style="font-size:10px;text-transform:uppercase;letter-spacing:0.7px;color:var(--muted);font-weight:600;margin-top:2px;">Conversion</div></div>
          <div style="flex:1;text-align:center;padding:14px;background:var(--surface);border-radius:8px;"><div style="font-family:'DM Serif Display',serif;font-size:28px;color:var(--muted);">{m["lost_qual"]}</div><div style="font-size:10px;text-transform:uppercase;letter-spacing:0.7px;color:var(--muted);font-weight:600;margin-top:2px;">Qualification</div></div>
        </div>
        <div style="background:var(--surface);border-radius:8px;padding:16px;font-size:12px;color:var(--muted);line-height:1.7;"><div style="color:var(--ink);font-weight:500;margin-bottom:4px;">Renewal tracking &mdash; Month 3 of Year 1</div>First renewals due Sep&ndash;Oct 2026. Churn rate measurable from Q4 2026.</div>
      </div>
    </div>
  </div>
</div>
<div class="section">
  <div class="section-header"><h2 class="section-title">Deals Needing Attention</h2><span class="section-note">Stuck &gt;30 days in current stage &middot; Top 20</span></div>
  <div class="card" style="padding:0;overflow:hidden;">
    <table class="flag-table"><thead><tr><th>School / Deal</th><th>Stage</th><th>Days in Stage</th><th>Value</th><th>Status</th></tr></thead><tbody>{render_stuck(m["stuck"])}</tbody></table>
  </div>
</div>
<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px 20px;font-size:11px;color:var(--muted);line-height:1.7;margin-bottom:40px;">
  <strong style="color:var(--ink);">Data:</strong> ActiveCampaign (ZAR &middot; SA only) &middot; Pipelines 4, 5, 6 &middot; UK/GBP excluded &middot; Renewal metrics activate Q4 2026 &middot; <strong style="color:var(--ink);">Updated:</strong> {TODAY}
</div>
</main>
<footer><span>the eAr Academy &middot; Sales Dashboard &middot; Confidential</span><span>Auto-generated {TODAY_ISO}</span></footer>
<script>
const leadsData={lm},wonData={wm},labels={lb};
function renderBars(id,data,color){{var el=document.getElementById(id);if(!el)return;var max=Math.max.apply(null,data.concat([1]));el.innerHTML=data.map(function(v,i){{var h=Math.max(2,Math.round(v/max*96));return'<div class="bar-wrap"><div class="bar" style="height:'+h+'px;background:'+color+';position:relative;"><span class="bar-val">'+(v||'')+'</span></div><span class="bar-label">'+labels[i]+'</span></div>';}}).join('');}}
renderBars('leadsChart',leadsData,'var(--lapis)');renderBars('wonChart',wonData,'var(--green)');
</script>
</body></html>"""


def main():
    print("\n   Ear Academy - Sales Dashboard Update")
    print("=" * 50)

    print("\n   Connecting to ActiveCampaign...")

    print("   Pipeline 4 - Qualification...")
    p4 = fetch_zar_deals(status=OPEN, group=P_QUAL)
    print(f"   -> {len(p4)} open ZAR deals")

    print("   Pipeline 5 - Conversion...")
    p5 = fetch_zar_deals(status=OPEN, group=P_CONV)
    print(f"   -> {len(p5)} open ZAR deals")

    print("   Pipeline 6 - CAM Won...")
    cam_won = fetch_zar_deals(status=WON, group=P_CAM)
    print(f"   -> {len(cam_won)} won ZAR deals")

    print("   Pipeline 6 - CAM Onboarding...")
    cam_open = fetch_zar_deals(status=OPEN, group=P_CAM)
    print(f"   -> {len(cam_open)} open ZAR deals")

    print("   Lost deals...")
    lost = fetch_zar_deals(status=LOST)
    print(f"   -> {len(lost)} lost ZAR deals")

    print("   SA campaigns...")
    campaigns = fetch_sa_campaigns()
    print(f"   -> {len(campaigns)} SA campaigns")

    print("\n   Calculating metrics...")
    m = calculate(p4, p5, cam_won, cam_open, lost, campaigns)
    print(f"   Customers: {m['customers_total']} | ARR: {m['arr_str']}")
    print(f"   Pipeline:  {m['pipeline_str']} | Coverage: {m['pipeline_coverage']}x")
    print(f"   New leads: {m['p4_this_month']} this month | Stuck >30d: {len(m['stuck'])}")

    print("\n   Building investor.html...")
    html = build_html(m)
    DASHBOARD.write_text(html, encoding="utf-8")
    print(f"   -> Saved")

    print("\n   Pushing to GitHub...")
    try:
        os.chdir(SCRIPT_DIR)
        subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", f"Update sales dashboard {TODAY_ISO}"],
            capture_output=True, text=True
        )
        if "nothing to commit" in result.stdout:
            print("   -> No changes to push (data unchanged)")
        else:
            subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
            print("   -> Pushed successfully")
        print(f"\n   Live at:")
        print("   https://earacademy.github.io/Ear-Academy-Usage-Tracker/investor.html\n")
    except subprocess.CalledProcessError as e:
        print(f"   Git error: {e}")
        print("   Dashboard rebuilt locally. Run git push manually.\n")

    print("Done!\n")


if __name__ == "__main__":
    main()
