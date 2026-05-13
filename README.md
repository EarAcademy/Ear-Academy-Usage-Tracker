# Ear Academy Analytics

Static-HTML dashboards for Ear Academy, deployed to GitHub Pages. Three Python scripts pull data from ActiveCampaign and weekly Excel snapshots, then commit updated HTML/JSON to this repo.

**Live site:** https://earacademy.github.io/Ear-Academy-Usage-Tracker/

## Daily update

```bash
bash ~/Desktop/ear-academy-analytics/update_all_dashboards.sh
```

Runs all three dashboard updaters in one go (sales → usage → velocity), then a single commit + push.

## Dashboards

| URL | What it shows | Updated by |
|---|---|---|
| `/index.html` | Usage Analytics + Usage Patterns (both tabs auto-generated) | `update_dashboard.py` |
| `/investor.html` | Sales & Growth Overview | `update_sales_dashboard.py` |
| `/pipeline_velocity.html` | Pipeline Velocity | `update_velocity.py` |

## Source of truth

The list of paying schools comes from **ActiveCampaign**, not from the Excel snapshots:

- `update_sales_dashboard.py` writes `paying_schools.json` — every deal in AC Pipeline 6 (Customer Account Management), stages Onboarding + Activated, ZAR currency, B2C accounts excluded.
- `update_dashboard.py` reads that JSON and treats it as the canonical list of paying schools. Any snapshot row whose school name resolves to a roster entry is counted; everything else is dropped.
- A small `SCHOOL_NAME_ALIASES` dict in `update_dashboard.py` reconciles snapshot ↔ AC name mismatches (e.g. snapshot says `dr.vanderross`, AC says `Dr. V.D.Ross - C5`).

This means the dashboard automatically tracks AC. Add a school to Pipeline 6 → it appears on the next run. Move one out → it disappears. No code edits needed.

## Full documentation

See [`CLAUDE.md`](./CLAUDE.md) for complete system documentation — architecture, data flows, schemas, credentials, troubleshooting, and due-diligence checklist.
