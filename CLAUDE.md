# Ear Academy Analytics — System Documentation

**Single source of truth for the Ear Academy public dashboards.** This document covers system architecture, data flows, scripts, operations, security, and handover procedures. Written to survive due-diligence review and onboard a new operator without external context.

- **Last reviewed:** 12 May 2026
- **System owner:** Rus Nerwich (rus@the-ear.com)
- **Repository:** `git@github.com:EarAcademy/Ear-Academy-Usage-Tracker.git`
- **Public site:** https://earacademy.github.io/Ear-Academy-Usage-Tracker/
- **Local working copy:** `~/Desktop/ear-academy-analytics/`

---

## 1. Purpose

The Ear Academy Analytics system maintains three public-facing dashboards that surface customer-usage and pipeline-health metrics for investors, the board, and the internal team. Updates are driven by a small set of Python scripts that pull from ActiveCampaign (CRM) and parse weekly Excel snapshots, then commit the output HTML/JSON to a public GitHub Pages site. Everything is automated by a single shell script run from the operator's Mac.

The system is intentionally simple: static HTML + JSON + a few Python scripts. There is no server, no database, no authentication layer. The only stateful infrastructure is the GitHub repository.

---

## 2. System Architecture

```
┌─────────────────────────┐        ┌─────────────────────────┐
│  ActiveCampaign API     │        │ Daily Excel Snapshots   │
│  (CRM — live data)      │        │ (login data from EarAc) │
└───────────┬─────────────┘        └───────────┬─────────────┘
            │                                  │
            ▼                                  ▼
   update_sales_dashboard.py            update_dashboard.py
   update_velocity.py                          │
            │                                  │
            ▼                                  ▼
   pipeline_data.json              ┌──→ index.html (rebuilt)
   velocity_data.json              │   (Usage Analytics +
            │                      │    Usage Patterns tabs)
            ▼                      │
   investor.html (reads JSON) ─────┘
   pipeline_velocity.html (reads JSON)
            │
            ▼
   git add + commit + push
            │
            ▼
   GitHub repo (origin/main)
            │
            ▼
   GitHub Pages auto-deploy (~60s)
            │
            ▼
   https://earacademy.github.io/Ear-Academy-Usage-Tracker/
```

**Three update paths, one site.** The Sales and Velocity dashboards are JSON-driven (Python writes JSON, HTML reads it on page load). The Usage dashboard is HTML-driven (Python rebuilds the entire index.html each run from the snapshot folder).

---

## 3. Repository & Hosting

- **Git repo:** `EarAcademy/Ear-Academy-Usage-Tracker` (private/public — verify in GitHub settings)
- **Remote:** `git@github.com:EarAcademy/Ear-Academy-Usage-Tracker.git` (SSH; requires a deploy key or operator's SSH key on the GitHub account)
- **Default branch:** `main`
- **Hosting:** GitHub Pages, serving from `main` root. Live URL is the GitHub Pages URL above.
- **Deployment latency:** ~30–120 seconds after `git push` for GitHub Pages to rebuild and serve.

---

## 4. File Inventory

All files live in `~/Desktop/ear-academy-analytics/`.

### Scripts (Python)

| File | Purpose | Writes to | Reads from |
|---|---|---|---|
| `update_dashboard.py` | Builds the Usage Analytics + Usage Patterns dashboard | `index.html` | `daily_snapshots/*.xlsx` |
| `update_sales_dashboard.py` | Pulls pipeline + email data from ActiveCampaign | `pipeline_data.json` | ActiveCampaign API |
| `update_velocity.py` | Pulls deal-stage data from ActiveCampaign | `velocity_data.json` | ActiveCampaign API |
| `add_nav.py` | One-off helper that injected the nav bar into `investor.html` (kept for reference) | `investor.html` | — |
| `config.py` | Holds `AC_API_KEY` and `AC_BASE_URL`. **Never commit real values.** | — | — |

### Scripts (shell)

| File | Purpose |
|---|---|
| `update_all_dashboards.sh` | Runs all three Python updaters in sequence, then a single git commit + push. **This is the daily-driver script.** |
| `update_and_publish.sh` | Older helper (predates `update_all_dashboards.sh`). Kept for reference. |

### Dashboards (HTML)

| File | What it shows | How it gets data |
|---|---|---|
| `index.html` | Usage Analytics tab (Daily Pulse, Weekly Snapshot, Patterns This Week, Weekly Trends) AND Usage Patterns tab (heatmap of school engagement Jan–Apr 2026) | Rebuilt in full by `update_dashboard.py` |
| `investor.html` | Sales Activity dashboard — pipeline funnel, monthly sales activity, email outreach metrics, ARR tiers | Fetches `pipeline_data.json` on load (client-side JavaScript) |
| `pipeline_velocity.html` | Pipeline Velocity — time-in-stage distribution, deals needing attention, sales funnel | Fetches `velocity_data.json` on load |
| `tam-depletion-prototype.html` | Static prototype of a TAM-depletion dashboard. Not currently maintained. |

### Data

| File / Folder | Contents | Updated by |
|---|---|---|
| `daily_snapshots/` | 80+ Excel files, one per day, exported manually from the Ear Academy platform. Naming: `Daily Usage Snapshot - DD-MM-YYYY.xlsx` (prefixes `B2C - `, `UK - ` allowed) | Operator drops new file in each day |
| `pipeline_data.json` | Current sales pipeline state — qualification/demo/negotiation counts, monthly leads, email outreach, ARR tiers | `update_sales_dashboard.py` |
| `velocity_data.json` | Pipeline 4 & 5 stage data — totals, average days in stage, time-in-stage buckets, stale deals | `update_velocity.py` |
| `archive/` | Older snapshots kept for historical reference | Manual |

### Documentation

| File | Purpose |
|---|---|
| `CLAUDE.md` (this file) | Complete system documentation |
| `README.md` | Project overview + quick-start (points to CLAUDE.md) |
| `DASHBOARD_SETUP_GUIDE.md` | Original setup guide from March 2026. Now superseded by this document. |

### Excluded / sensitive

| File | Notes |
|---|---|
| `config.py` | Contains live ActiveCampaign credentials. **Must remain in `.gitignore`.** If it has been accidentally committed, rotate the AC API key immediately. |
| `.git/` | Standard Git metadata |
| `__pycache__/` | Python bytecode cache. Safe to delete. |

---

## 5. Daily Workflow

### Standard daily routine (operator)

1. Download the day's usage snapshot from the Ear Academy platform.
2. Save it to `~/Desktop/ear-academy-analytics/daily_snapshots/` (filename `Daily Usage Snapshot - DD-MM-YYYY.xlsx`).
3. Open Terminal and run:
   ```
   bash ~/Desktop/ear-academy-analytics/update_all_dashboards.sh
   ```
4. Wait for `SUMMARY` block. Each of the three dashboards shows ✅ or ❌.
5. Hard-refresh the three live URLs (Cmd+Shift+R) to verify GitHub Pages has rebuilt.

### What the script does

1. Runs `update_dashboard.py` → rebuilds `index.html`.
2. Runs `update_sales_dashboard.py` → rewrites `pipeline_data.json`.
3. Runs `update_velocity.py` → rewrites `velocity_data.json`.
4. Single `git add -A` → `git commit -m "Update dashboards [...] YYYY-MM-DD HH:MM"` → `git push origin main`.
5. Prints a summary indicating which dashboards updated successfully.

### Failure handling

- If one Python script fails, the others still run. The summary block reports failures.
- A failed dashboard means the source file (HTML or JSON) is untouched — no corruption, no half-state.
- If git push fails (e.g., auth/network), the local commits remain and can be retried with `git push origin main`.

---

## 6. Dashboards in Detail

### Usage Analytics (index.html — primary tab)

Visible at https://earacademy.github.io/Ear-Academy-Usage-Tracker/index.html

Sections:
- **Daily Pulse:** yesterday's login counts (classroom vs instrumental), trend vs prior day, new schools today
- **Weekly Snapshot:** week-over-week comparison, schools activated lifetime/this week, consistency
- **Patterns This Week:** new activations, contextual notes
- **Weekly Trends:** 6-week chart of classroom vs instrumental usage

Data source: parsed from all `*.xlsx` files in `daily_snapshots/`. The script segments by:
- **Product Type:** "Classroom", "Classroom & Instrumental", "Instrumental"
- **Billing Status:** Pilot/Demo schools excluded from primary metrics
- **Exclusion list:** Specific schools (Academie Orpheus, C5, Bolton Music Services, etc.) hardcoded in the script

### Usage Patterns (index.html — second tab)

Same HTML file, separate tab. Shows the Jan–Apr 2026 engagement heatmap with school classifications (Power User, High-frequency, Consistent, Bi-weekly, Early stage, One-time, Gone quiet).

Both tabs update simultaneously every time `update_dashboard.py` runs — there is no separate "usage patterns" script.

### Sales Activity (investor.html)

Visible at https://earacademy.github.io/Ear-Academy-Usage-Tracker/investor.html

Sections:
- Key numbers (active customers, ARR, total pipeline value, new leads MTD)
- Sales pipeline funnel (qualification → demo → negotiation → won)
- Monthly sales activity (new leads, new deals, demos done by month)
- Email outreach (sent / opened / open rate)
- 6-month new ZAR lead trend

Data source: ActiveCampaign API, fetched live by `update_sales_dashboard.py`. Pulls counts from Pipelines 4 (Sales Qualification), 5 (Sales Conversion), 6 (Customer Account Management).

### Pipeline Velocity (pipeline_velocity.html)

Visible at https://earacademy.github.io/Ear-Academy-Usage-Tracker/pipeline_velocity.html

Sections:
- Stage cards (New Leads, Demo/Pilot, Negotiation) with totals, average days in stage, total value
- Complete Sales Funnel — ZAR Deals
- "What this means for investors" — conversion ratios with targets
- Time-in-Stage distribution by stage
- Deals needing attention (stuck > 60 days)

Data source: ActiveCampaign API. Pipelines 4 & 5 only (Pipeline 6 customer-management is excluded from velocity tracking).

---

## 7. Script Reference

### update_dashboard.py

- **Inputs:** All `.xlsx` files in `daily_snapshots/`
- **Outputs:** `index.html` (full rebuild)
- **Frequency:** Daily (after a new snapshot is added)
- **Side effects:** None — pure read of snapshots, pure write of HTML

Key behaviours:
- Filename parsing regex: `(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{4})` — tolerates variants like `DD-MM-YYYY`, `DD - MM - YYYY`, prefixes `B2C - `, `UK - `, suffixes `(1)`
- Default segment when older snapshots lack Product Type / Billing Status columns: assumes "Paying + Classroom & Instrumental"
- Excludes schools defined in `EXCLUDED_SCHOOLS` list and any with Billing Status of "Pilot" or "Demo"
- Hardcoded constants: `WEEK1_START = 2026-01-19`, `TOTAL_CUSTOMERS = 53` (auto-updated in main)

### update_sales_dashboard.py

- **Inputs:** ActiveCampaign API
- **Outputs:** `pipeline_data.json`
- **Frequency:** Weekly (or anytime a fresh sales snapshot is wanted)
- **Side effects:** Reads from AC; no writes to AC

Pipeline IDs queried:
- `P_QUAL = "4"` — Sales Qualification
- `P_CONV = "5"` — Sales Conversion
- `P_CAM = "6"` — Customer Account Management
- `S_NEW_LEAD = "36"` — Stage in Pipeline 4

ZAR deals only (USD/GBP deals are filtered out).

### update_velocity.py

- **Inputs:** ActiveCampaign API
- **Outputs:** `velocity_data.json`
- **Frequency:** Weekly
- **Side effects:** Read-only on AC. Explicitly does **not** touch `pipeline_velocity.html`.

Tracks Pipelines 4 & 5 only. Stage IDs:
- `STAGE_NEW_LEAD = 36`
- `STAGE_DEMO = 43` (with historical aliases 13, 26, 44, 45)
- `STAGE_NEGO = 46` (with historical aliases 27, 47)

Outputs include per-pipeline totals, average days in stage, total ZAR value, time-in-stage buckets (u30 / u60 / u90 / u180 / over180), and stale-deal listings.

### update_all_dashboards.sh

The daily-driver. Runs all three Python updaters with independent error handling, then a single git commit+push containing whatever changed. Designed so one failure does not block the others.

---

## 8. Data Schemas

### pipeline_data.json

```json
{
  "pipeline":  { "qualification": int, "demo": int, "negotiation": int, "customers": int },
  "monthly":   { "jan|feb|mar": { "new_leads": str, "new_deals": str } },
  "email":     { "sent": str, "replied": str, "rate": str (e.g. "74.9%"), "new_contacts": str },
  "arr_tiers": { "tier1..tier4": { "label": str, "count": int, "revenue": float } },
  "lost_deals": { ... }
}
```

### velocity_data.json

```json
{
  "generated_at": ISO-8601 timestamp,
  "generated_label": human-readable date,
  "pipeline4": {
     "total": int,
     "avg_days_in_stage": int,
     "total_value_zar": int,
     "buckets": { "u30": int, "u60": int, "u90": int, "u180": int, "over180": int },
     "stale_deals": [ { "id": str, "title": str, "days_in_stage": int, "total_age_days": int } ]
  },
  "pipeline5": {
     "demo":  { ... same shape as pipeline4 ... },
     "nego":  { ... },
     "total_in_conversion": int
  },
  "conversion_rates": { "lead_to_demo_pct": float, "demo_to_nego_pct": float },
  "stuck_summary":    { "conversion_over_60": int, "conversion_over_90": int, "new_leads_over_180": int, "all_stages_over_90": int }
}
```

---

## 9. Credentials & Security

### What's sensitive

- **`AC_API_KEY`** in `config.py` — full read access to the ActiveCampaign account
- **GitHub SSH key** on the operator's machine — push access to the public repo

### Where credentials live

| Credential | Location | Access |
|---|---|---|
| AC API key | `config.py` on operator's Mac only | Should be in `.gitignore` — confirm with `git check-ignore config.py` |
| AC base URL | `config.py` | Same |
| GitHub auth | SSH key in `~/.ssh/` | Tied to GitHub account |

### Rotation procedure

1. In ActiveCampaign: Settings → Developer → generate a new API key
2. Update `config.py` on the operator's Mac
3. Test with: `python3 update_velocity.py` (read-only call)
4. Revoke the old key in ActiveCampaign
5. If the old key was ever committed: rotate **immediately** and search the repo history with `git log -p config.py` to confirm exposure scope

### Risk register

- **Single operator** — if Rus's Mac is unavailable, no one else can update. Mitigation: document the AC API key + GitHub deploy-key in a password manager accessible to the board.
- **No automated testing** — script failures only surface when run. Mitigation: monitor the public site for a stale "Last updated" timestamp.
- **GitHub Pages public** — every commit and historical version is publicly visible. Anything sensitive must never enter this repo.

---

## 10. ActiveCampaign Reference

### Pipeline structure

| Pipeline ID | Name | Purpose | Used by |
|---|---|---|---|
| 4 | Sales Qualification | New leads, demo prep | sales + velocity |
| 5 | Sales Conversion | Demo/Pilot, Negotiation | sales + velocity |
| 6 | Customer Account Management | Onboarding, active customers | sales (count only) |

### Stage IDs (canonical + historical)

| Stage | Canonical ID | Historical IDs (deduped in script) |
|---|---|---|
| New Lead | 36 | — |
| Demo / Pilot | 43 | 44, 45, 13, 26 |
| Negotiation | 46 | 47, 27 |
| Onboarding | 25 | 50 |
| Activated | 51 | — |

Currency filter: scripts only count ZAR deals (USD/GBP excluded).

---

## 11. Common Tasks (How-to)

### Add a new daily snapshot
Drop the `.xlsx` into `daily_snapshots/` with the filename pattern. Run `update_all_dashboards.sh`.

### Re-run a single dashboard
```bash
cd ~/Desktop/ear-academy-analytics
python3 update_dashboard.py        # Usage
python3 update_sales_dashboard.py  # Sales
python3 update_velocity.py         # Velocity
git add -A && git commit -m "..." && git push origin main
```

### Add a school to the exclusion list
Edit `EXCLUDED_SCHOOLS` in `update_dashboard.py` (around line 25), commit, push.

### Add a new dashboard
1. Create a new HTML file in the folder
2. Either (a) write a new `update_X.py` that produces a JSON the HTML reads, or (b) extend `update_dashboard.py` to write a new section
3. Add the new HTML to the navigation in existing dashboards
4. Add the new updater call to `update_all_dashboards.sh`
5. Commit and push

### Investigate a discrepancy between dashboards and the truth
1. Check `pipeline_data.json` / `velocity_data.json` timestamps to confirm last update
2. Compare counts to the ActiveCampaign UI directly
3. If numbers diverge, the issue is most likely a stage-ID mismatch (a new stage was added in AC that the scripts don't yet handle). Look up the new stage ID and add it to `STAGE_NAMES` in `update_velocity.py`.

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Dashboard shows old date | Browser cache | Cmd+Shift+R hard refresh; if still old, check that `git push` succeeded |
| `git push` fails | SSH key not loaded, or no network | `ssh-add ~/.ssh/id_*` then retry; check internet |
| `❌ Could not find config.py` | Running script from wrong directory | `cd ~/Desktop/ear-academy-analytics` first |
| `'requests' library not found` | Python package not installed | `pip3 install requests --break-system-packages` |
| Snapshot date not picked up | Filename doesn't match the regex | Rename to `Daily Usage Snapshot - DD-MM-YYYY.xlsx` |
| One dashboard "verify failed" in cleanup | Cloud metadata size mismatch (false positive) | Check file counts match; size differences are normal in synced folders |
| New stage in AC shows wrong name | Stage ID not in `STAGE_NAMES` lookup | Add the ID to the `STAGE_NAMES` dict in `update_velocity.py` |

---

## 13. Glossary

- **AC** — ActiveCampaign, the CRM
- **ARR** — Annual Recurring Revenue
- **Pipeline 4 / 5 / 6** — ActiveCampaign sales pipelines (Qualification / Conversion / Customer Management)
- **ZAR / GBP / USD deals** — Deal currency (only ZAR counted in current dashboards)
- **Power User / Hi-freq / Consistent / Bi-weekly / Early stage / One-time / Gone quiet** — Usage Patterns engagement classifications used in index.html
- **Time-in-stage bucket** — Number of days a deal has been in its current pipeline stage (u30 = under 30, etc.)
- **Pilot / Demo school** — Non-paying schools excluded from primary metrics

---

## 14. Due-Diligence Checklist

For someone evaluating or taking over the system:

- [ ] Verify access to the GitHub repo (`EarAcademy/Ear-Academy-Usage-Tracker`)
- [ ] Verify GitHub Pages is serving from `main` root and the site loads
- [ ] Verify the operator's machine has a working `config.py` with valid AC credentials (run `python3 update_velocity.py` — should write `velocity_data.json` without errors)
- [ ] Confirm `config.py` is in `.gitignore` and not in the repo history (`git log --all -- config.py` should return empty)
- [ ] Cross-check 5 random metrics on the public dashboards against the same numbers in ActiveCampaign's UI
- [ ] Read the last 30 days of git commit messages — should be a clean cadence of `Update dashboards [...]` commits
- [ ] Verify `daily_snapshots/` has files reasonably up to date (most-recent snapshot should be ≤ 3 working days old)
- [ ] Test a full run end-to-end: drop a fresh snapshot, run `update_all_dashboards.sh`, confirm dashboards refresh
- [ ] Confirm at least one other person knows where the AC API key is stored
- [ ] Confirm GitHub repo collaborator list matches who is supposed to have access

---

## 15. Change Log

| Date | Change |
|---|---|
| 2026-05-12 | Removed orphan `1index.html` (legacy backup, not part of any workflow). Commit `677ce7d`. |
| 2026-05-12 | Created CLAUDE.md (this document). Migrated folder from `analytics-dashboard` back to `ear-academy-analytics`. Added `update_all_dashboards.sh` to run all three updaters in one command. |
| 2026-05-11 | Migrated dashboard folder from Desktop to OneDrive and back during a Desktop cleanup. |
| 2026-05-08 | Last reliable index.html update before migration. |
| Earlier | See `git log` for full commit history. |

---

## 16. Contacts

- **Primary operator:** Rus Nerwich (rus@the-ear.com)
- **Repository owner (GitHub org):** EarAcademy
- **CRM admin (ActiveCampaign):** [TODO — confirm and add]

---

*End of document. Keep this file in the repository root and update it whenever scripts, schemas, or workflows change.*
