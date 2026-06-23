# Retention Insights — Setup Checklist for the New Company Claude Account

Use this to confirm everything has come across correctly when you start using the company Claude account instead of your personal one. Work through it top to bottom — each step says what to check and why it matters.

## 1. Confirm the files themselves moved over

This whole project is just files — there's no separate "Claude project" or database to migrate, only what's sitting in this folder. Check that this `Retention Insights` folder, wherever it now lives (presumably the same SharePoint/OneDrive location, just accessed from the company account), contains all of these:

- [ ] `Retention Insights.html` — the dashboard itself
- [ ] `CLAUDE.md` — project context file
- [ ] `DATA-REFERENCE.md` — the data dictionary
- [ ] `IT-CTO-INFRASTRUCTURE.md` — the doc for your Head of IT/CTO
- [ ] `PROJECT-SETUP-CHECKLIST.md` — this file
- [ ] `source-data/` folder, containing the original Term 1 file and the Term 2 check-in report

If any are missing, it's almost certainly because the folder sync (OneDrive/SharePoint) hasn't finished, or the folder was copied rather than moved. Re-copy the whole `Retention Insights` folder rather than picking individual files — the documents reference each other and are meant to travel together.

**Why this matters:** unlike a typical app, there's nothing else to "install." If the files are present, the project is present.

## 2. Confirm the dashboard still opens and works

- [ ] Double-click `Retention Insights.html` (or open it via your browser's File > Open). It should open directly — no login screen, no loading spinner that hangs.
- [ ] Confirm you see 5 stat cards at the top (Check-ins Logged, At Risk, Actively Engaged, Most Common Signal, Latest Reply Rate).
- [ ] Confirm the "School Trend — Across Terms" table shows two term columns (Term 1 2026 and Term 2 2026) and a Trend column on the right.
- [ ] Scroll to the log table at the bottom and confirm you can filter by Term, Category, and Status.

If all of this looks the way it did before the migration, the file itself is intact and nothing was corrupted in the move.

## 3. Connect this folder in the company Cowork account

When you start a new chat in the company account, you'll need to select (mount) the folder containing this project, the same way you did on your personal account. Once selected, Claude will be able to read and edit these files directly, the same way it does now.

- [ ] Start a new chat on the company account and select the folder that contains `Retention Insights` (or the `Retention Insights` folder itself, if you'd rather scope access tightly to just this project).

## 4. Check that Claude picks up the project context automatically

`CLAUDE.md` exists specifically so a brand-new chat — one with no memory of this conversation — can still understand the project immediately, without you having to re-explain it.

- [ ] In a fresh chat on the company account (with the folder selected), ask something like: *"What is the Retention Insights project, and what's the one rule I need to follow before adding new school data?"*
- [ ] Claude should be able to answer correctly (mentioning the exact-school-name-matching rule) **without you uploading or re-pasting anything** — it should find this from `CLAUDE.md` and `DATA-REFERENCE.md` directly in the folder.

If it can't answer correctly, paste this checklist's folder path into the chat and ask Claude to read `CLAUDE.md` directly — that always works as a fallback, it just means the automatic pickup isn't happening for some reason.

**Why this matters:** this is the single most important thing this documentation set is solving for. Conversation history does not transfer between Claude accounts — everything Claude "remembers" about this project from your old account's chats is gone on the new one. `CLAUDE.md` and `DATA-REFERENCE.md` are what replace that lost memory, so this check confirms the replacement is actually working.

## 5. Hand the infrastructure doc to your Head of IT / CTO

- [ ] Share `IT-CTO-INFRASTRUCTURE.md` with them directly (email attachment, or a link to this folder if they have access). It explains what this dashboard is, where its data lives, what's sensitive about it, and what access controls are recommended — written so they don't need any Claude-specific context to understand it.

## 6. Confirm access/sharing permissions

- [ ] Make sure the SharePoint/OneDrive folder this lives in is shared only with the people who actually need it (you, and now your Head of IT/CTO) — not a public or "anyone with the link" share. See Section 5 of `IT-CTO-INFRASTRUCTURE.md` for the reasoning.

## 7. You're done when...

All seven boxes above are checked, and you've successfully had a fresh-chat conversation on the company account that correctly understood the project without you re-explaining it from scratch. At that point, this migration is complete and you can continue adding Term 3 data exactly the way Term 2 was added.
