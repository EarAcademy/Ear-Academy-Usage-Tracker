# Ear Academy Analytics

Static-HTML dashboards for Ear Academy, deployed to GitHub Pages. Three Python scripts pull data from ActiveCampaign and weekly Excel snapshots, then commit updated HTML/JSON to this repo.

**Live site:** https://earacademy.github.io/Ear-Academy-Usage-Tracker/

## Daily update

```bash
bash ~/Desktop/ear-academy-analytics/update_all_dashboards.sh
```

Runs all three dashboard updaters in one go, then a single commit + push.

## Dashboards

| URL | What it shows | Updated by |
|---|---|---|
| `/index.html` | Usage Analytics + Usage Patterns | `update_dashboard.py` |
| `/investor.html` | Sales & Growth Overview | `update_sales_dashboard.py` |
| `/pipeline_velocity.html` | Pipeline Velocity | `update_velocity.py` |

## Full documentation

See [`CLAUDE.md`](./CLAUDE.md) for complete system documentation — architecture, data flows, schemas, credentials, troubleshooting, and due-diligence checklist.
