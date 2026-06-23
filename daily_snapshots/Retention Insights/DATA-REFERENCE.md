# Retention Insights — Data Reference

This is the data dictionary for `Retention Insights.html`. Check this file before adding, editing, or asking Claude to add any new school entry — most data errors in this project so far have come from typing a school's name slightly differently than it appears elsewhere in the data.

## Canonical school name list

These are the **exact spellings** currently used in the dashboard, verified directly from the live file as of this document's creation (22 entries logged, 17 distinct schools, Term 1 2026 and Term 2 2026). Always copy a school's name from this list rather than retyping it from memory or from a new source document.

| Canonical Name | Contact (most recent) | Current Category | Current Status | Last Checked |
|---|---|---|---|---|
| Calling Education | Martin Botha | No Response | At Risk | Term 2 2026 |
| CBC Mount Edmund | KG | Assessment / Test Period | Monitoring | Term 1 2026 |
| Curro Langebaan | Carien van der Walt | Infrastructure Disruption | Monitoring | Term 2 2026 |
| Danie Ackermann Primary | Pearl Ackerman | No Response | At Risk | Term 1 2026 |
| Deutsche Schule Hermannsburg | Gerard du Toit | Active & Engaged | Engaged | Term 2 2026 |
| Herzlia Highlands Primary | Sandy Segal | Product-Fit Friction | Pending | Term 2 2026 |
| Herzlia Weizmann | Albert Karating | No Response | At Risk | Term 2 2026 |
| Holy Cross RC Primary | Nomthandazo Zweni (via Chandre Dias & Arlene Thomas) | Active & Engaged | Engaged | Term 2 2026 |
| Lebone College | Mutubatsa Tshuma | No Response | At Risk | Term 2 2026 |
| Pinelands High School | Eloise Jurgens | Curriculum Conflict | Monitoring | Term 1 2026 |
| Plettenberg Bay Christian School | Ayden Kroukamp | No Response | At Risk | Term 2 2026 |
| Prescient Primary | Roxanne | Assessment / Test Period | Monitoring | Term 2 2026 |
| St Catherine's | Lisa Parkin | Product-Fit Friction | Monitoring | Term 2 2026 |
| St Martin Prep | Kim (Forgus K) | No Response | At Risk | Term 2 2026 |
| Sun Valley Primary | Karla van Niekerk | Product-Fit Friction | Monitoring | Term 2 2026 |
| Trinity House | Ana Silent | No Response | At Risk | Term 2 2026 |
| Unknown / Direct Contact | Siyabonga (siyabonga991709@gmail.com) | No Response | Pending | Term 2 2026 |

**Important — there are 3 real Herzlia schools, only 2 are tracked here.** Herzlia Highlands Primary (Sandy Segal) and Herzlia Weizmann (Albert Karating) are two separate, distinct schools and must never be merged into one row. A third Herzlia school exists in Ear Academy's customer base but had no check-in data in Term 1 or Term 2, so it has no row yet. If you log data for it, give it its own distinct, correctly-spelled name — do not reuse "Herzlia Highlands Primary" or "Herzlia Weizmann."

**Open item — "Unknown / Direct Contact" needs identity resolution.** This entry covers a Gmail contact (siyabonga991709@gmail.com) who did not reply to the Term 2 check-in. There's a reasonable chance this is the same person as Siyabonga Motloung, the new, actively-engaged teacher at Deutsche Schule Hermannsburg — in which case this row should be deleted (he's already tracked, and engaged, under Hermannsburg) rather than carried forward as a separate at-risk/pending school. Resolve this before Term 3 data entry if possible.

## Category definitions (`cat` field)

| id | Label | Meaning |
|---|---|---|
| `active-engaged` | Active & Engaged | Using the platform with no significant friction — a positive signal, not a problem to solve. |
| `curriculum` | Curriculum Conflict | Platform competes with other subjects/rotations for class time. |
| `assessment` | Assessment / Test Period | Teacher paused for cycle tests, exams, or end-of-term events. Expected to self-resolve next term. |
| `technical` | Technical Barrier | Login issues, forgotten credentials, device failure, setup friction. |
| `infrastructure` | Infrastructure Disruption | Renovations, no classroom, no computer access — a physical/logistical blocker, not a product issue. |
| `product-fit` | Product-Fit Friction | The platform's structure doesn't match how this particular teacher wants to work (e.g. wants topic-based search instead of weekly plans). Often comes with constructive, positive feedback attached. |
| `no-response` | No Response | School hasn't replied to the check-in. Highest-risk category — unknown reason. |
| `other` | Other / Contextual | One-off or seasonal reasons not covered above. |

## Status definitions (`status` field)

| id | Label | Meaning | Score (for trend calc) |
|---|---|---|---|
| `at-risk` | At Risk | Active concern, needs follow-up. | 1 |
| `pending` | Pending | Awaiting a reply or a planned next step. | 2 |
| `monitoring` | Monitoring | Aware of an issue, watching, not yet resolved. | 3 |
| `resolved` | Resolved | An issue existed and has been fixed. | 4 |
| `engaged` | Engaged | Actively using the platform well, no open issues. | 5 |

The "Score" column is only used internally by `computeTrend()` to decide whether a school's flag should read Improving, Declined, or Stable — it compares the score of a school's earliest logged term against its latest. It is not shown anywhere in the UI.

## Term format

Always `Term N YYYY` — e.g. `Term 1 2026`, `Term 2 2026`, `Term 3 2026`. The space before the year and the capitalization of "Term" don't strictly matter (the parser is case-insensitive and tolerant of extra spaces), but for consistency, always type it exactly as `Term N YYYY`.

## Changelog of data corrections

Keep this section updated any time a naming or categorization issue is found and fixed — the entire point of this dashboard is the trend view, and that breaks silently (no error message) if a school's name isn't spelled identically across terms.

- **Hermannsburg.** Term 1 originally used "Hermannsburg School." Term 2's check-in report used "Deutsche Schule Hermannsburg" — the fuller, correct name. Term 1's entries were updated to match.
- **Herzlia Highlands.** Term 1 originally used "Herzlia Highlands." Term 2's check-in report used "Herzlia Highlands Primary" — the fuller, correct name. Term 1's entries were updated to match.
- **Herzlia Middle School → Herzlia Weizmann.** The Term 2 check-in report's summary table named Albert Karating's school "Herzlia Middle School." Rus confirmed this was incorrect — Albert Karating is at Herzlia Weizmann, a separate school from Sandy Segal's Herzlia Highlands Primary. Corrected in the data on 22 June 2026.
- **St Catherine's — follow-up reply (22 June 2026).** St Catherine's was originally logged as "No Response / At Risk" in the initial Term 2 round. Lisa Parkin then replied directly: she also uses other musical resources in class, so her usage of Ear Academy is intermittent by nature — not disengagement. Logged as a second Term 2 2026 entry (`product-fit` / `monitoring`), which supersedes the earlier no-response record since it has a later date. The original no-response entry was left in place rather than deleted, for an accurate history of the back-and-forth.
- **Holy Cross RC Primary recategorization.** The Term 2 check-in report's summary table listed Holy Cross under "Assessment Period," but the detailed write-up for that school describes active, spreading adoption (two grades using it, a refresher session planned) — a positive engagement story, not a test-period pause. Logged as `active-engaged` / `engaged` instead of the report's literal category. Flagged transparently to Rus; not yet independently reconfirmed against the report author's original intent.
