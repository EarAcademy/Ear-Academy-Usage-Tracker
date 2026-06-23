# Retention Insights — Infrastructure & Data Documentation

Prepared for: Head of IT / CTO, The Ear Academy
Prepared: 22 June 2026
Owner/business contact: Rus Nerwich (rus@the-ear.com)

## 1. What this is

Retention Insights is an internal dashboard that tracks school engagement check-ins over time — which schools are at risk of churning, which are actively engaged, and why, term over term. It is used by Rus to prioritize outreach and to surface investor-shareable engagement signals. It is not customer-facing and is not part of the Ear Academy product — it's an internal operations tool.

## 2. Architecture — there isn't much of one, by design

This is a **single static HTML file** (`Retention Insights.html`, ~45KB) with HTML, CSS, and JavaScript all embedded in that one file. There is:

- No server
- No database
- No build process or compiler
- No third-party JavaScript libraries or frameworks
- No CDN-hosted resources
- No API calls of any kind, to Ear Academy systems or anyone else's
- No network activity whatsoever — confirmed by inspecting the file for any `http://` or `https://` references; there are none

Opening the file in any modern web browser (Chrome, Safari, Edge, Firefox) runs the entire application client-side, with no internet connection required after the file itself is obtained.

**Why this matters for IT:** there is no hosting to provision, no domain, no SSL certificate, no server patching, no uptime to monitor, and no attack surface beyond "someone with the file can read/edit it locally." This is an appropriate level of infrastructure for the current data volume (~20 records) and team size (effectively single-user). Section 7 covers what would need to change if that stops being true.

## 3. Where the data lives and how it flows

All check-in records are **hardcoded as a JavaScript array inside the file itself** — there is no external data source the dashboard reads from at runtime. When Rus (or whoever maintains this) wants to add a new term's data, the array inside the file is edited directly and a new copy of the file is saved.

There is a secondary, **lower-reliability** data path: the dashboard has an on-page form that lets a user add an entry without editing code. Entries added this way are saved to the browser's `localStorage` under the key `ear-retention-insights`. This is **local to that one browser, on that one device** — it does not sync to OneDrive/SharePoint, does not appear if the file is opened on a different computer, and does not appear for a different person opening the same file. This is a known, accepted limitation (not a bug) for the current single-user usage pattern, but it is the single biggest data-integrity risk in this design if usage patterns change — see Section 7.

**No data leaves the device.** Nothing is transmitted anywhere. The file is fully self-contained.

## 4. Data sensitivity

The dashboard contains:

- School names
- Named individual contacts at each school (teachers, principals, administrators)
- Some personal email addresses (e.g., a Gmail address used by one contact)
- Free-text notes describing each school's situation, including direct quotes from contacts

This is **business contact data, not student data** — no learner names, no student records, no data covered by school-specific student-privacy obligations appears anywhere in this file. The relevant sensitivity is closer to standard CRM/sales-contact data: it would be mildly embarrassing or could damage a school relationship if it leaked publicly (e.g., a contact's email being spammed, or a school's internal friction being made visible to that school), but it is not regulated health/financial/minor data.

**Recommendation:** treat this file with the same access discipline as your sales CRM export — private SharePoint/OneDrive folder, not a publicly-shared link, not committed to a public code repository, not attached to an email distribution wider than necessary.

## 5. Access control

There is no login, password, or permission system built into the file itself — anyone who can open it can read and edit all of it. Access control is entirely a function of **where the file is stored**, not the file itself.

**Current recommendation:** keep this file (and the rest of this folder) in a private, permissions-restricted SharePoint or OneDrive folder, shared only with the people who need it (currently: Rus, and presumably yourself for this migration). Do not place it in a publicly accessible location, a public GitHub repo, or a public web host (e.g., GitHub Pages) without first stripping out the contact names and emails — doing so would expose real people's names and email addresses to anyone with the link.

## 6. Backup, versioning, and the Claude account migration

This file has no built-in version history. Whatever backup/versioning exists is whatever the storage location provides — e.g., SharePoint/OneDrive version history, if the folder is stored there.

This documentation set was generated using Claude (Anthropic's AI assistant) in "Cowork" mode, which had read/write access to a folder Rus selected on his own computer (via a synced SharePoint/OneDrive folder) during the session. Claude does not retain this data anywhere outside that session and outside the files written to that folder — there is no persistent Claude-hosted copy of this data once the session ends. The file and its data exist only in: (a) wherever this folder is stored/synced on Ear Academy's infrastructure, and (b) transiently, within whichever AI session is actively being used to edit it.

As part of migrating from a personal to a company Claude account, this project folder (see structure below) should simply be carried over as files — there is no Claude-side database, project, or configuration that needs separate migration. See `PROJECT-SETUP-CHECKLIST.md` for the practical steps.

## 7. Known limitations and when to revisit this architecture

This single-file approach is an intentional, appropriate choice for the current scale (~20 schools, effectively one person maintaining it). It should be revisited if any of the following becomes true:

- **More than one person needs to add/edit data regularly.** The hardcoded-array + localStorage-fallback model has no real multi-user support — two people editing independently will not see each other's changes and could overwrite each other's work if both produce a new version of the file. At that point, a shared backend (even something lightweight like a Google Sheet or Airtable read via API, or a small database-backed internal tool) would remove the risk of silently lost data.
- **The data volume grows substantially** (hundreds of schools, multiple check-ins per term). The current design re-renders everything client-side from a single in-memory array; it will continue to work but editing the hardcoded array by hand becomes increasingly error-prone at scale.
- **The data needs to be queried or reported on outside this dashboard** (e.g., pulled into a BI tool, joined with the Sales Pipeline CRM data). At that point it would make sense to treat this dashboard as a "view" on top of a proper data store rather than the data store itself.

None of these conditions currently apply, so no architecture change is recommended at this time — flagging them here so they're a deliberate, informed decision later rather than something discovered the hard way.

## 8. Quick technical summary

| Property | Value |
|---|---|
| File type | Single `.html` file, self-contained |
| File size | ~45 KB |
| Backend | None |
| Database | None |
| External dependencies | None (zero network calls, zero CDN references) |
| Browser compatibility | Any modern browser (Chrome, Safari, Edge, Firefox); uses standard HTML5/CSS3/ES6 JavaScript only |
| Data storage | Hardcoded JS array (primary, portable) + browser `localStorage` (secondary, device-local only) |
| Authentication | None — access controlled entirely via file storage location/permissions |
| Network activity | None |
| Personal data present | Business contact names and emails for ~17 schools; no student data |
