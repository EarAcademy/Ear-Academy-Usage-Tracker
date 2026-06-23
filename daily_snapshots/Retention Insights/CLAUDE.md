# Retention Insights — Project Memory

> Read this file first, before touching any other file in this folder. It tells you what this project is, how its data is structured, and which mistakes have already been made and fixed once — don't repeat them.

## What this is

**Retention Insights** is a single-file HTML dashboard built for The Ear Academy to track school check-in outcomes across terms — both problems (technical barriers, no response, product-fit friction) and positive signals (active engagement, testimonials). It replaced an earlier, narrower tool called "inactivity-tracker.html" that only covered Term 1 and only tracked at-risk schools.

Owner: Rus Nerwich (rus@the-ear.com), founder of The Ear Academy.

There is no backend, no build step, no install process. The whole thing is one HTML file with embedded CSS and JavaScript. Opening it in any browser (double-click, or open via File > Open) runs the full dashboard.

## Files in this folder

| File | Purpose |
|---|---|
| `Retention Insights.html` | The dashboard itself. This is the only file that needs to be opened to use the tool. |
| `CLAUDE.md` | This file — project context for any AI assistant picking up this project. |
| `DATA-REFERENCE.md` | Data dictionary: canonical school names, category/status definitions, term format, and a changelog of data corrections already made. **Read this before adding or editing any school entry.** |
| `IT-CTO-INFRASTRUCTURE.md` | Technical/infrastructure documentation written for the Head of IT and CTO — architecture, data storage, security, and access-control recommendations. |
| `PROJECT-SETUP-CHECKLIST.md` | A checklist for confirming this project is correctly set up after moving to a new computer or Claude account. |
| `source-data/` | The original Term 1 dashboard file and the Term 2 check-in report (.docx) that the current dashboard's hardcoded data was built from. Kept for traceability — if a number in the dashboard looks wrong, this is where to verify it against the original source. |

## How the data works

All check-in data lives **hardcoded inside the `<script>` block** of `Retention Insights.html`, in a JavaScript array called `entries`. This is a deliberate choice, not an oversight: the dashboard also has an on-page "+ Log Entry" form that writes to the browser's `localStorage`, but `localStorage` is local to one browser on one device. If data were only added through the form, it would not travel with the file across SharePoint/OneDrive/other devices. Hardcoding into the file itself is what makes this dashboard portable and reliable across the team.

**Practical implication:** when a new term's check-in round is complete, the recommended way to add it is to ask Claude to hardcode the new entries directly into the `entries` array (matching the existing pattern), not to rely solely on typing them into the on-page form. The form is fine for one-off, single-user, single-device additions, but the source of truth for anything that needs to be shared or to survive across devices is the hardcoded array.

Each entry has this shape:

```js
{ id: 23, school: 'Exact Canonical Name', contact: 'Contact Person', cat: 'category-id',
  status: 'status-id', notes: '...', action: '...', date: 'YYYY-MM-DD', term: 'Term 3 2026' }
```

See `DATA-REFERENCE.md` for the full list of valid `cat` and `status` values, and the canonical spelling of every school currently tracked.

## The single most important rule: exact school name matching

The cross-term trend table groups entries by an **exact string match** on `school`. If the same school is spelled two different ways across two terms (e.g. "Hermannsburg School" vs "Deutsche Schule Hermannsburg"), the trend table will silently treat them as two different schools and the whole point of the tool — seeing how a school's status changes over time — breaks, with no error or warning.

This has already happened twice in this project's history (see `DATA-REFERENCE.md` changelog). Before adding a new entry for a school that already exists in the data, copy its exact spelling from `DATA-REFERENCE.md` rather than retyping it from memory or from a new source document.

## How terms work

Terms are plain strings in the format `Term N YYYY` (e.g. `Term 2 2026`, `Term 3 2026`). There is no hardcoded list of terms anywhere — the dashboard derives the list of terms, the term filter dropdown, the trend table's columns, and the "next term" suggestion in the add-entry form all automatically from whatever term strings exist in the data, via a regex (`/Term\s*(\d+)\s*(\d{4})/i`). This means adding Term 3 data does not require editing any function — it is picked up automatically as soon as entries with `term: 'Term 3 2026'` exist in the array.

## Common tasks

- **Add a new term's check-in results:** hardcode new objects into the `entries` array, following the existing pattern and using exact canonical school names from `DATA-REFERENCE.md`. Update the changelog in `DATA-REFERENCE.md` if any new name variants or corrections come up.
- **Add a new category or status:** add to the `CATEGORIES` array or `STATUS_META`/`STATUS_SCORE` objects near the top of the `<script>` block, plus a matching CSS class if it's a new status (follow the `.status-engaged` pattern), plus add it to the `<select>` option lists in the HTML (`filter-status`, `f-status`).
- **Check the file still works after an edit:** there is no automated test suite. At minimum, open the file in a browser and confirm the stat cards, trend table, and log table all render without a blank/broken section. For a more thorough check, a future Claude session can extract the `<script>` block and run `node --check` on it for a syntax check, and use `jsdom` to render it headlessly and verify entry counts.

## What this dashboard deliberately does NOT do

- No server, no database, no API calls, no analytics tracking, no external dependencies of any kind.
- No multi-user real-time sync. If two people open the file in different browsers and both add entries through the on-page form, neither will see the other's additions — only hardcoded entries in the file itself are guaranteed to be shared.
- No authentication. Anyone who can open the file can see and edit the data.

These are intentional simplicity tradeoffs for a small team, not bugs. See `IT-CTO-INFRASTRUCTURE.md` for the reasoning and for what would need to change if the team or data sensitivity grows.
