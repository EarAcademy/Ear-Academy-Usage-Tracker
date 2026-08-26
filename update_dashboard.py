#!/usr/bin/env python3
"""
Ear Academy Usage Analytics - Dashboard Updater
Segments data by Product Type and Billing Status.
- Paying schools: all primary metrics
- Demo schools:   UK Pilot tile
- Classroom:      Product Type in ("Classroom", "Classroom & Instrumental")
- Instrumental:   Product Type == "Instrumental"
Old files (no Product Type / Billing Status cols) default to Paying + Classroom & Instrumental.
"""

import difflib
import json
import pandas as pd
import re
import unicodedata
from collections import Counter
from pathlib import Path
from datetime import datetime, timedelta


def strf(dt, fmt):
    """Cross-platform strftime. glibc/BSD support '%-d' (day, no leading zero)
    but Windows' C runtime rejects it with ValueError. Substitute the numeric
    day/month/hour before delegating so the same format strings work on Mac,
    Linux, and Windows.
    """
    if '%-d' in fmt:
        fmt = fmt.replace('%-d', str(dt.day))
    if '%-m' in fmt:
        fmt = fmt.replace('%-m', str(dt.month))
    if '%-I' in fmt:
        fmt = fmt.replace('%-I', str((dt.hour - 1) % 12 + 1))
    return dt.strftime(fmt)


# ── Configuration ────────────────────────────────────────────────────────────
DATA_FOLDER     = Path("daily_snapshots")
OUTPUT_FILE     = Path("index.html")
ROSTER_FILE     = Path("paying_schools.json")    # written by update_sales_dashboard.py
REPORT_FILE     = Path("daily_report.txt")       # human-readable "what happened" log
WEEK1_START     = datetime(2026, 1, 19)
TOTAL_CUSTOMERS = 49          # set from AC roster in main() — fallback to 49

# Schools to HIDE from the Usage dashboard even though they appear in the AC
# roster (paying_schools.json). The roster is regenerated from ActiveCampaign
# on every run, so removing a school here — not by editing the JSON — is what
# makes the removal stick across runs. Matched case- and accent-insensitively
# against each roster entry's title AND account_name.
#
# NOTE: this only affects the USAGE dashboard (index.html). These deals still
# exist in ActiveCampaign and still count on the Sales dashboard. To remove a
# school everywhere, its deal must be moved out of AC Pipeline 6 as well.
ROSTER_EXCLUDE = {
    "cheré botha school",
    "raff home school",
    "windemere primary",
    "xero test",
}


def _norm_name(s):
    """Lowercase + strip accents, for robust name comparison (so 'Cheré'
    matches 'chere' and casing/accents can't cause a miss)."""
    s = unicodedata.normalize('NFKD', str(s or '')).encode('ascii', 'ignore').decode('ascii')
    return s.strip().lower()


def _is_roster_excluded(entry):
    """True if this roster entry's title or account_name is in ROSTER_EXCLUDE."""
    norm_excl = {_norm_name(x) for x in ROSTER_EXCLUDE}
    return (_norm_name(entry.get('title')) in norm_excl
            or _norm_name(entry.get('account_name')) in norm_excl)

# Structured record of what each run loaded / skipped, so the operator can see
# exactly why a file did or didn't make it onto the dashboard. Populated by
# load_all_data(), consumed by the end-of-run report in main().
_LOAD_REPORT = {
    'loaded':               [],   # (filename, date, week)
    'skipped_no_date':      [],   # filename — no DD-MM-YYYY in the name
    'skipped_before_wk1':   [],   # filename — dated before WEEK1_START
    'skipped_no_sheet':     [],   # filename — no usable data sheet
    'skipped_missing_cols': [],   # filename — no School Name / Email column
    'errors':               [],   # (filename, error message)
    'header_typos':         [],   # (filename, canonical column name, actual header found)
    'roster_error':         None, # reason paying_schools.json failed to load, or None if fine
}

# ── Tolerant column matching ──────────────────────────────────────────────────
# A header typo (e.g. "Emai Address" instead of "Email Address") used to drop
# the ENTIRE file silently, because the exact-substring check found nothing.
# This fuzzy fallback catches near-miss headers instead. It requires both a
# high absolute similarity AND a clear margin over every other column in the
# same file, so it won't guess between two genuinely different columns (e.g.
# "Full Name" vs "School Name" both containing "Name") — tested against the
# real column set to confirm the margin holds (best ~0.95 vs runner-up ~0.60).
# An unclear case is left unmatched (file dropped, as before) rather than
# risking a wrong column being silently treated as School/Email.
HEADER_FUZZY_THRESHOLD = 0.80
HEADER_FUZZY_MARGIN    = 0.15


def find_column(columns, required_substrings, canonical_label):
    """Find a column by exact substring match first; falls back to fuzzy
    matching against `canonical_label` (e.g. "Email Address") if nothing
    matches exactly. Returns (column_or_None, matched_via_fuzzy: bool).
    """
    exact = next((c for c in columns
                  if all(s in str(c).lower() for s in required_substrings)), None)
    if exact is not None:
        return exact, False

    scored = sorted(
        ((c, difflib.SequenceMatcher(None, canonical_label.lower(), str(c).lower()).ratio())
         for c in columns),
        key=lambda t: -t[1])
    if not scored:
        return None, False
    best_col, best_score = scored[0]
    runner_up_score = scored[1][1] if len(scored) > 1 else 0.0
    if best_score >= HEADER_FUZZY_THRESHOLD and (best_score - runner_up_score) >= HEADER_FUZZY_MARGIN:
        return best_col, True
    return None, False

EXCLUDED_SCHOOLS = [
    'Academie Orpheus',
    'Academie Orfeus',
    'Bolton Music Services',
    'Bradford Music and Arts Service',
    'Bury Music',
    'Collingwood College',
    'Salford Community Leisure'
]

# Snapshot names we already KNOW are not ZAR paying customers (UK music services,
# pilots, one-off logins). They will always be dropped by the roster gate — that's
# correct. Listing them here just keeps them out of the "investigate" bucket in the
# daily report so a genuine miss (e.g. a real school not yet in Pipeline 6) stands
# out. Purely cosmetic for the report; does NOT affect what the dashboard counts.
KNOWN_NON_PAYING = {
    'academie orpheus', 'academie orfeus', 'bolton music services',
    'bradford music and arts service', 'bury music', 'collingwood college',
    'salford community leisure', 'oldham music service',
    'beckfoot priestthorpe school',
    # NOTE: 'range high school' was removed — it is a South African school, not a
    # UK music service. It has snapshot usage but no ActiveCampaign deal, so it
    # belongs in INVESTIGATE (a real school that may need a deal created) rather
    # than being hidden as a known non-paying exclusion.
}

# UK pilot schools — these are NOT ActiveCampaign Pipeline-6 paying customers,
# so they never appear on the main (paying) dashboard. They ARE tracked on their
# own "UK Pilots" tab. Matched case-insensitively against the snapshot School
# name. Identified by their UK email domains (.gov.uk / .co.uk / UK academy).
# Edit this list to add or remove a UK pilot.
UK_PILOT_SCHOOLS = {
    'bolton music services',
    'bradford music and arts service',
    'bury music',
    'oldham music service',
    'salford community leisure',
    'beckfoot priestthorpe school',
}


def should_exclude_school(school_name, billing_status=None):
    """Return True if school should be excluded from dashboard."""
    if billing_status in ['Pilot', 'Demo']:
        return True
    school_lower = school_name.lower()
    for excluded in EXCLUDED_SCHOOLS:
        excl_lower = excluded.lower()
        if len(excl_lower) <= 3:
            # Short names: exact match only to avoid false partial matches
            if school_lower == excl_lower:
                return True
        else:
            if excl_lower in school_lower:
                return True
    return False

# ── Canonical name overrides ──────────────────────────────────────────────────
EXACT_OVERRIDES = {
    'Acudeo Thornview Primary &amp; Secondary School': 'Acudeo Thornview',
    'Acudeo Thornview Primary & Secondary School':     'Acudeo Thornview',
    "St Martin&#039;s Preparatory Schoo":              'St Martin Preparatory School',
    "St Martin's Preparatory School":                  'St Martin Preparatory School',
    "St Martin&#039;s Preparatory School":             'St Martin Preparatory School',
}

# ── Snapshot ↔ AC roster name aliases ────────────────────────────────────────
# Maps snapshot school names that DON'T auto-match the AC roster
# (paying_schools.json) to a string that DOES match an AC roster entry's
# `title` or `account_name`. Everything else auto-matches and needs no entry
# here. Keep this list as the single place to reconcile name mismatches.
SCHOOL_NAME_ALIASES = {
    # Snapshot name                       →  Matches in AC roster (title or account_name)
    "Acudeo Protea Glen":                   "Acudeo College Protea Glen",
    "Applewood Preparatory":                "Applewood Preparatory School",
    "CBC Mount Edmund":                     "CBC Mount Edmund (Christian Brothers' College Mount Edmund)",
    "Educ8sa":                              "Educ8 SA",
    "dr.vanderross":                        "Dr. V.D.Ross - C5",
    "Harriston Primary School":             "Harriston School (Primary)",
    "Hermannsburg School":                  "Hermannsburg School (Primary)",
    "Herzlia High School":                  "Herzlia Renewal 2025",
    "Herzlia Highlands":                    "Herzlia Primary",
    "Herzlia Weitzman Primary School":      "Herzlia Weitzman",
    "Holy Cross RC Primary":                "Holy Cross R C Primary",
    "Lebone II College":                    "Lebone II College (Primary)",
    "Princess Park College":                "Royal Schools Princess Park",
    "Sky City Primary School":              "Royal Schools Sky City",
    "St Catherines":                        "ST CATHERINE'S DOMINICAN CONVENT- SA",
    "St Martin Preparatory School":         "St Martin's Preparatory School",
    "St Martins Preparatory School":        "St Martin's Preparatory School",
    "Sunvalley Primary School":             "Sun Valley",
    "Trinity House":                        "TrinityHouse",
}


# ── Display-name overrides ────────────────────────────────────────────────────
# Purely cosmetic: controls the label a school SHOWS UNDER on the dashboard.
# By default a school displays under whatever name appears most often in the
# daily snapshots — which is sometimes an ugly system string (e.g. "Educ8sa",
# "dr.vanderross"). Map "what currently shows" → "what to show instead" here.
#
# This does NOT affect matching or which schools count — only the visible label.
# To tidy a name: run the dashboard, see how it appears in the report / page,
# and add that exact current label on the left with the desired label on the right.
DISPLAY_NAME_OVERRIDES = {
    # Currently shows as   →  Show as instead
    "Educ8sa":               "Educ8 SA",
    "dr.vanderross":         "Dr. V.D. Ross",
}


def apply_display_override(name):
    """Return the preferred display label for a school name, if one is set."""
    return DISPLAY_NAME_OVERRIDES.get(name, name)


# ── Paying-schools roster (loaded from paying_schools.json) ──────────────────
# Populated by main() at startup. _ROSTER is the list, _ROSTER_LOOKUP is the
# lowercased-name → roster-entry dict used by resolve_to_roster().
# _DISPLAY_NAME_FOR_DEAL maps each AC deal_id to the canonical display name
# (the most-common snapshot label observed). Used by paying() to normalise
# the School column so groupby/nunique can't double-count name variants.
_ROSTER                 = None
_ROSTER_LOOKUP          = None
_DISPLAY_NAME_FOR_DEAL  = None


def load_paying_schools_roster():
    """Load paying_schools.json. Returns (roster_list, lookup_dict).
    Returns (None, None) if file is missing or invalid — caller should then
    fall back to legacy (snapshot-only) mode.
    """
    if not ROSTER_FILE.exists():
        _LOAD_REPORT['roster_error'] = f"{ROSTER_FILE} does not exist"
        return None, None
    try:
        with open(ROSTER_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ⚠️  Could not load {ROSTER_FILE}: {e}")
        _LOAD_REPORT['roster_error'] = (
            f"{ROSTER_FILE} is corrupted/truncated and could not be parsed as JSON: {e}. "
            f"Likely an interrupted write by update_sales_dashboard.py (network drop, "
            f"laptop sleep, or crash mid-write). Re-run update_sales_dashboard.py, or "
            f"restore the file from the last good git commit.")
        return None, None

    roster = data.get("schools", []) or []
    if not roster:
        _LOAD_REPORT['roster_error'] = f"{ROSTER_FILE} parsed OK but contains zero schools"
        return None, None

    # Drop any manually-excluded schools (ROSTER_EXCLUDE) — hidden from the
    # usage dashboard even though AC still lists them. Done here so it survives
    # every roster regeneration.
    if ROSTER_EXCLUDE:
        before = len(roster)
        roster = [e for e in roster if not _is_roster_excluded(e)]
        removed = before - len(roster)
        if removed:
            print(f"  🚫  ROSTER_EXCLUDE: hid {removed} school(s) from the dashboard "
                  f"({before} → {len(roster)})")

    # Index every roster entry by its lowercased title AND account_name.
    # First-wins on collisions so the title takes priority over account_name.
    lookup = {}
    for entry in roster:
        for key_field in ("title", "account_name"):
            key = (entry.get(key_field) or "").strip().lower()
            if key and key not in lookup:
                lookup[key] = entry

    # Apply explicit aliases on top: snapshot_name → matching roster entry
    for snap_name, target in SCHOOL_NAME_ALIASES.items():
        target_low = (target or "").strip().lower()
        snap_low   = (snap_name or "").strip().lower()
        if target_low in lookup and snap_low:
            lookup[snap_low] = lookup[target_low]

    # Make display-override LABELS resolvable too. paying() renames a school's
    # rows to its display name, and that name is later re-resolved (e.g. in the
    # patterns tab). If the override label isn't a lookup key it falls back to
    # fuzzy matching every run (noisy + fragile) — so register each override
    # label against the same entry its current name resolves to.
    for shown_name, pretty in DISPLAY_NAME_OVERRIDES.items():
        shown_low  = (shown_name or "").strip().lower()
        pretty_low = (pretty or "").strip().lower()
        if shown_low in lookup and pretty_low and pretty_low not in lookup:
            lookup[pretty_low] = lookup[shown_low]

    return roster, lookup


def resolve_to_roster(name, lookup=None):
    """Return the roster entry for a snapshot school name, or None.

    Tries an exact match first (fast path, and always preferred). If that
    fails, falls back to a fuzzy (token-overlap) match against every roster
    title/account_name — this is what catches the day-to-day spelling drift
    between the Ear Academy platform export and ActiveCampaign (a missing
    apostrophe, "&" vs "and", a dropped "Primary", etc.) so a school with
    real logins doesn't silently vanish from the dashboard just because its
    name isn't byte-for-byte identical to AC's.

    Fuzzy hits require both a high absolute score AND a clear margin over
    the next-best *distinct* school, so it won't confidently guess between
    two similarly-named schools — an unclear case is left unmatched (and
    shows up in the daily report) rather than silently mis-attributed.
    Every fuzzy hit is cached (stable for the rest of this run) and recorded
    in _FUZZY_MATCH_LOG for the report, so it can be reviewed and promoted
    to an explicit SCHOOL_NAME_ALIASES entry if correct.
    """
    lookup = lookup if lookup is not None else _ROSTER_LOOKUP
    if not lookup or not name:
        return None
    key = str(name).strip().lower()
    if not key:
        return None

    exact = lookup.get(key)
    if exact is not None:
        return exact

    if key in _FUZZY_CACHE:
        return _FUZZY_CACHE[key]

    entry = _fuzzy_resolve_roster(key, lookup)
    _FUZZY_CACHE[key] = entry
    if entry is not None:
        _FUZZY_MATCH_LOG[str(name).strip()] = clean_roster_display_name(entry)
    return entry


FUZZY_MATCH_THRESHOLD = 0.72   # minimum token-overlap score to consider a hit
FUZZY_MATCH_MARGIN    = 0.15   # winner must beat the next distinct school by this much

_FUZZY_CACHE     = {}   # snapshot name (lowered) -> roster entry or None, memoised per run
_FUZZY_MATCH_LOG = {}   # snapshot name (as seen)  -> matched display name, for the report


def _fuzzy_resolve_roster(key, lookup):
    """Best fuzzy match for `key` among all lookup keys, keyed by distinct
    school (deal_id) so two aliases of the SAME school don't look like a
    false 'ambiguous' runner-up. Returns the roster entry, or None if no
    candidate clears both the score threshold and the safety margin.
    """
    best_by_deal = {}
    for cand_key, entry in lookup.items():
        score = _token_overlap(key, cand_key, stopwords=_ROSTER_STOPWORDS)
        if score <= 0:
            continue
        deal_id = entry.get('deal_id')
        if score > best_by_deal.get(deal_id, (0.0, None))[0]:
            best_by_deal[deal_id] = (score, entry)

    if not best_by_deal:
        return None

    ranked = sorted(best_by_deal.values(), key=lambda t: -t[0])
    best_score, best_entry = ranked[0]
    runner_up_score = ranked[1][0] if len(ranked) > 1 else 0.0

    if best_score >= FUZZY_MATCH_THRESHOLD and (best_score - runner_up_score) >= FUZZY_MATCH_MARGIN:
        return best_entry
    return None


def clean_roster_display_name(entry):
    """Friendly display name for a roster entry — used for silent schools."""
    title = (entry.get("title") or "").strip()
    for suffix in (" - Core Education Group", "-Core Education Group",
                   " - Core Education",       "-Core Education"):
        if title.endswith(suffix):
            title = title[:-len(suffix)].strip()
            break
    if title.isupper():
        title = title.title()
    return title


def build_display_name_for_deal(combined_df):
    """For each roster deal_id, pick the most-common snapshot name seen for it.
    This becomes the canonical display name used everywhere on the dashboard,
    so a single school can't appear twice under spelling variants.
    """
    if _ROSTER_LOOKUP is None or combined_df is None or combined_df.empty:
        return {}
    rows = combined_df[['School']].dropna().copy()
    rows['DealId'] = rows['School'].apply(
        lambda n: (resolve_to_roster(n) or {}).get('deal_id'))
    rows = rows.dropna(subset=['DealId'])
    if rows.empty:
        return {}
    counts = (rows.groupby(['DealId', 'School']).size()
                   .reset_index(name='c')
                   .sort_values(['DealId', 'c'], ascending=[True, False]))
    best = counts.drop_duplicates('DealId').set_index('DealId')['School'].to_dict()
    # Apply cosmetic display-name overrides (e.g. "Educ8sa" → "Educ8 SA").
    return {did: apply_display_override(name) for did, name in best.items()}

MERGE_BLOCKLIST = {
    'Bay Primary',
    'Plettenberg Bay Christian Primary School',
}

# ── Fuzzy-matching helpers ────────────────────────────────────────────────────

def _clean_for_matching(s):
    s = s.replace('\xa0', ' ').replace('\u200b', '').replace('\u2019', "'")
    s = s.replace('&amp;', '&').replace('&#039;', "'").replace('&apos;', "'")
    s = s.replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii').lower()
    s = re.sub(r"['\-\u2013\u2014,.]", ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\bprepatory\b', 'preparatory', s)
    s = re.sub(r'\bst\b',        'saint',        s)
    s = re.sub(r'\bschoo\b',     'school',       s)
    return s


# Words stripped when consolidating spelling VARIANTS of the same known school
# (build_canonical_map, below). Loose on purpose — at that point every input is
# already confirmed to be one school, so over-matching risk is low.
_MERGE_STOPWORDS = {'school', 'primary', 'secondary', 'college', 'academy',
                    'the', 'of', 'and', 'saint', 'high', 'preparatory'}

# Words stripped when matching a snapshot name against the DISTINCT roster of
# different schools (_fuzzy_resolve_roster, below). Deliberately narrower:
# 'primary' / 'secondary' / 'high' / 'preparatory' are kept as real tokens
# here because they're often the only thing distinguishing two campuses of
# the same name-family (e.g. "Herzlia High School" vs "Herzlia Primary" are
# different accounts) — treating them as noise caused a real false match in
# testing. Only genuinely institution-generic words are dropped.
_ROSTER_STOPWORDS = {'school', 'college', 'academy', 'the', 'of', 'and', 'saint'}


def _token_overlap(a, b, stopwords=None):
    if stopwords is None:
        stopwords = _MERGE_STOPWORDS
    ta = set(_clean_for_matching(a).split()) - stopwords or set(_clean_for_matching(a).split())
    tb = set(_clean_for_matching(b).split()) - stopwords or set(_clean_for_matching(b).split())
    if not ta or not tb:
        return 0.0
    inter  = len(ta & tb)
    union_ = len(ta | tb)
    return inter / union_ if union_ else 0.0


def build_canonical_map(raw_names, threshold=0.80):
    from collections import Counter
    freq   = Counter(raw_names)
    unique = list(freq.keys())
    parent = {n: n for n in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, a in enumerate(unique):
        for b in unique[i+1:]:
            if a in MERGE_BLOCKLIST or b in MERGE_BLOCKLIST:
                continue
            if _token_overlap(a, b) >= threshold:
                union(a, b)

    groups = {}
    for name in unique:
        groups.setdefault(find(name), []).append(name)

    canonical_map = {}
    for members in groups.values():
        canonical = max(members, key=lambda n: (freq[n], len(n)))
        for m in members:
            canonical_map[m] = canonical
    return canonical_map


def normalize_school_name(name, canonical_map=None):
    if pd.isna(name):
        return ""
    s = str(name).replace('\xa0', ' ').replace('\u200b', '').replace('\u2019', "'").strip()
    s = EXACT_OVERRIDES.get(s, s)
    if canonical_map and s in canonical_map:
        s = canonical_map[s]
    return s


# ── Product type helpers ──────────────────────────────────────────────────────

def classify_product(pt):
    """Return 'classroom', 'instrumental', or 'both' from a raw Product Type value."""
    if pd.isna(pt):
        return 'both'          # old files default → counts in both buckets
    pt = str(pt).strip().lower()
    if 'classroom' in pt and 'instrumental' in pt:
        return 'both'
    if 'classroom' in pt:
        return 'classroom'
    if 'instrumental' in pt:
        return 'instrumental'
    return 'both'              # unknown → both


def classify_billing(bs):
    """Return 'Paying', 'Demo', or 'Paying' (default for old files).
    Both 'Demo' and 'Pilot' billing statuses are treated as non-paying (mapped to 'Demo').
    """
    if pd.isna(bs):
        return 'Paying'
    s = str(bs).strip()
    if s in ('Demo', 'Pilot'):
        return 'Demo'
    return s if s == 'Paying' else 'Paying'


# ── Date / week helpers ───────────────────────────────────────────────────────

def parse_date(filename):
    m = re.search(r'(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{4})', filename)
    if not m:
        return None
    day, month, year = m.groups()
    try:
        return datetime(int(year), int(month), int(day))
    except ValueError:
        return None


def assign_week(date):
    if date < WEEK1_START:
        return None
    return (date - WEEK1_START).days // 7 + 1


def week_label(week_num):
    start = WEEK1_START + timedelta(weeks=week_num - 1)
    end   = start + timedelta(days=4)
    if start.month == end.month:
        return f"Week {week_num} ({strf(start, '%-d')}–{strf(end, '%-d %b')})"
    return f"Week {week_num} ({strf(start, '%-d %b')}–{strf(end, '%-d %b')})"


def pct_change_html(new_val, old_val):
    if old_val == 0:
        return '<span class="delta new">new</span>'
    diff = new_val - old_val
    pct  = round((diff / old_val) * 100)
    if diff > 0:
        return f'<span class="delta up">▲ +{pct}%</span>'
    if diff < 0:
        return f'<span class="delta down">▼ {abs(pct)}%</span>'
    return '<span class="delta flat">→ same</span>'


# ── Data loading ──────────────────────────────────────────────────────────────

def find_data_sheet(sheet_names):
    """Return the best sheet name to use for raw data.
    Priority: 'Raw Data' (SA files) → 'Sheet1' (UK files) → first sheet.
    """
    for s in sheet_names:
        if 'Raw Data' in s:
            return s
    if 'Sheet1' in sheet_names:
        return 'Sheet1'
    return sheet_names[0] if sheet_names else None


def load_all_data():
    excel_files = sorted(DATA_FOLDER.glob("*.xlsx"))
    if not excel_files:
        return None

    # Pass 1 – build canonical name map
    raw_name_pool = []
    for file_path in excel_files:
        file_date = parse_date(file_path.name)
        if not file_date or assign_week(file_date) is None:
            continue
        try:
            xl    = pd.ExcelFile(file_path)
            sheet = find_data_sheet(xl.sheet_names)
            if not sheet:
                continue
            df  = pd.read_excel(file_path, sheet_name=sheet)
            col = next((c for c in df.columns
                        if 'school' in str(c).lower() and 'name' in str(c).lower()), None)
            if col:
                for v in df[col].dropna().unique():
                    raw_name_pool.append(EXACT_OVERRIDES.get(str(v).strip(), str(v).strip()))
        except Exception:
            pass

    canonical_map = build_canonical_map(raw_name_pool, threshold=0.80)

    merges = {c: vs for c, vs in
              {canon: [r for r, cc in canonical_map.items() if cc == canon and r != canon]
               for canon in set(canonical_map.values())}.items() if vs}
    if merges:
        print("\n  🔀 Name merges applied:")
        for canon, variants in sorted(merges.items()):
            for v in variants:
                print(f"      '{v}'  →  '{canon}'")
        print()
    else:
        print("  ✅ All school names consistent.\n")

    # Pass 2 – load rows with segmentation columns
    rows = []
    for file_path in excel_files:
        file_date = parse_date(file_path.name)
        if not file_date:
            print(f"  ⚠️  Skipped (no date): {file_path.name}")
            _LOAD_REPORT['skipped_no_date'].append(file_path.name)
            continue

        week = assign_week(file_date)
        if week is None:
            print(f"  ⚠️  Skipped (before Week 1): {file_path.name}")
            _LOAD_REPORT['skipped_before_wk1'].append(file_path.name)
            continue

        try:
            xl    = pd.ExcelFile(file_path)
            sheet = find_data_sheet(xl.sheet_names)
            if not sheet:
                print(f"  ⚠️  No usable sheet found: {file_path.name}")
                _LOAD_REPORT['skipped_no_sheet'].append(file_path.name)
                continue

            df = pd.read_excel(file_path, sheet_name=sheet)

            school_col,  school_fuzzy  = find_column(df.columns, ['school', 'name'], 'School Name')
            email_col,   email_fuzzy   = find_column(df.columns, ['email'],          'Email Address')
            product_col, _             = find_column(df.columns, ['product'],        'Product Type')
            billing_col, _             = find_column(df.columns, ['billing'],        'Billing Status')
            role_col,    _             = find_column(df.columns, ['role'],           'UserRole')

            for matched_col, was_fuzzy, canonical in (
                (school_col, school_fuzzy, 'School Name'),
                (email_col,  email_fuzzy,  'Email Address'),
            ):
                if was_fuzzy:
                    print(f"  🔤 Header typo caught in {file_path.name}: "
                          f"'{matched_col}' read as {canonical}")
                    _LOAD_REPORT['header_typos'].append((file_path.name, canonical, matched_col))

            if not school_col or not email_col:
                print(f"  ⚠️  Missing core columns: {file_path.name}")
                _LOAD_REPORT['skipped_missing_cols'].append(file_path.name)
                continue

            df = df.copy()
            df['School']   = df[school_col].apply(lambda n: normalize_school_name(n, canonical_map))
            df['Email']    = df[email_col]
            df['Date']     = file_date
            df['Week']     = week
            df['Product']  = df[product_col].apply(classify_product) if product_col else 'both'
            df['Billing']  = df[billing_col].apply(classify_billing) if billing_col else 'Paying'
            df['UserRole'] = df[role_col].astype(str).str.strip() if role_col else ''

            # Expand 'both' rows into two rows: one classroom, one instrumental
            # so downstream groupby works cleanly.
            # ProductExplicit=True  → file had a real Product Type column
            # ProductExplicit=False → file had no column; 'both' is a fallback, not real data
            has_product_col = product_col is not None
            expanded = []
            for _, row in df.iterrows():
                if row['Product'] == 'both':
                    expanded.append({**row, 'Product': 'classroom',    'ProductExplicit': has_product_col})
                    expanded.append({**row, 'Product': 'instrumental', 'ProductExplicit': has_product_col})
                else:
                    expanded.append({**dict(row), 'ProductExplicit': True})
            df = pd.DataFrame(expanded)

            # Remove internal rows
            mask = ~df['School'].str.contains(
                'Onboarding|Ear Academy|Knowledge Hub', case=False, na=False)
            df = df[mask & (df['School'] != '') & (df['School'] != 'nan')]

            rows.append(df[['School', 'Email', 'Date', 'Week', 'Product', 'Billing', 'UserRole', 'ProductExplicit']])

            # Count unique login rows (before expansion) for display
            orig_count = df.groupby(['School', 'Email', 'Date']).ngroups
            print(f"  ✓ {file_date.strftime('%a %d %b')}  (Week {week})  – {file_path.name}")
            _LOAD_REPORT['loaded'].append((file_path.name, file_date, week))

        except Exception as e:
            print(f"  ⚠️  Error reading {file_path.name}: {e}")
            _LOAD_REPORT['errors'].append((file_path.name, str(e)))
            import traceback; traceback.print_exc()

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'])
    return combined


# ── Metric calculations ───────────────────────────────────────────────────────

def paying(df):
    """Snapshot rows for schools that appear in the AC paying-schools roster.

    With roster (preferred): a row counts as 'paying' iff its School name
    resolves to a roster entry — the AC pipeline is the single source of
    truth, so EXCLUDED_SCHOOLS / billing-status filters become irrelevant.

    Also normalises the School column to the canonical display name for that
    roster entry (the most-common snapshot label observed). This means two
    snapshot spellings of the same school — e.g. 'St Martins Preparatory
    School' + 'St Martin Preparatory School' — collapse to a single entry
    in every downstream groupby/nunique calculation.

    Without roster (fallback for first-run / missing JSON): use the legacy
    rules — Billing == 'Paying' and not in EXCLUDED_SCHOOLS.
    """
    if _ROSTER_LOOKUP is None:
        pay = df[df['Billing'] == 'Paying']
        return pay[~pay['School'].apply(should_exclude_school)]

    in_roster = df['School'].apply(lambda n: resolve_to_roster(n) is not None)
    pay = df[in_roster].copy()

    if _DISPLAY_NAME_FOR_DEAL:
        def _canon(n):
            entry = resolve_to_roster(n)
            if not entry:
                return n
            return _DISPLAY_NAME_FOR_DEAL.get(entry['deal_id'], n)
        pay['School'] = pay['School'].apply(_canon)
    return pay


def demo(df):
    """Demo / Pilot rows used by the UK Pilot tile.
    Excludes rows that resolve to the paying roster (those are real customers,
    not pilots) when the roster is loaded.
    """
    if _ROSTER_LOOKUP is None:
        return df[df['Billing'] == 'Demo']
    not_roster = df['School'].apply(lambda n: resolve_to_roster(n) is None)
    return df[not_roster & (df['Billing'] == 'Demo')]

def classroom(df):
    return df[df['Product'] == 'classroom']

def instrumental(df):
    return df[df['Product'] == 'instrumental']

def unique_logins(df):
    """
    Count unique login events. Since 'both' rows are expanded to 2 rows,
    we deduplicate by (School, Email, Date) within each product segment.
    For total logins we use the original unique (School, Email, Date) count
    across either product.
    """
    return df.drop_duplicates(subset=['School', 'Email', 'Date', 'Product'])

def total_logins(df):
    """Unique (School, Email, Date) regardless of product expansion."""
    return df.drop_duplicates(subset=['School', 'Email', 'Date'])


def calc_daily_pulse(combined):
    pay = paying(combined)
    all_dates = sorted(pay['Date'].dt.date.unique())
    if not all_dates:
        return {}

    yesterday  = all_dates[-1]
    day_before = all_dates[-2] if len(all_dates) >= 2 else None

    y_df  = pay[pay['Date'].dt.date == yesterday]
    db_df = pay[pay['Date'].dt.date == day_before] if day_before else pay.iloc[0:0]

    # Logins = unique (School, Email, Date) events
    y_logins  = total_logins(y_df)['Email'].count()
    y_schools = y_df['School'].nunique()
    db_logins  = total_logins(db_df)['Email'].count()
    db_schools = db_df['School'].nunique() if day_before else 0

    # Product breakdown (deduplicated per product)
    y_cls  = unique_logins(classroom(y_df))['Email'].count()
    y_ins  = unique_logins(instrumental(y_df))['Email'].count()

    # New schools vs day before
    y_schools_set  = set(y_df['School'].unique())
    db_schools_set = set(db_df['School'].unique()) if day_before else set()
    new_today      = sorted(y_schools_set - db_schools_set)

    return {
        'yesterday':      yesterday,
        'day_before':     day_before,
        'y_logins':       y_logins,
        'y_schools':      y_schools,
        'y_cls':          y_cls,
        'y_ins':          y_ins,
        'db_logins':      db_logins,
        'db_schools':     db_schools,
        'new_schools':    new_today,
    }


def calc_weekly_snapshot(combined):
    pay       = paying(combined)
    max_week  = int(pay['Week'].max())
    prev_week = max_week - 1

    cw = pay[pay['Week'] == max_week]
    pw = pay[pay['Week'] == prev_week] if prev_week >= 1 else pay.iloc[0:0]

    # Total logins (no double-count from product expansion)
    cw_logins  = total_logins(cw)['Email'].count()
    cw_schools = cw['School'].nunique()
    pw_logins  = total_logins(pw)['Email'].count()
    pw_schools = pw['School'].nunique()

    # Product breakdown this week
    cw_cls_logins  = unique_logins(classroom(cw))['Email'].count()
    cw_ins_logins  = unique_logins(instrumental(cw))['Email'].count()
    cw_cls_schools = classroom(cw)['School'].nunique()
    cw_ins_schools = instrumental(cw)['School'].nunique()

    # Schools ever activated (paying)
    ever_schools     = pay['School'].nunique()
    prev_ever        = pay[pay['Week'] <= prev_week]['School'].nunique() if prev_week >= 1 else 0
    activated_change = ever_schools - prev_ever

    # Consistent: 3+ distinct days this week
    cw_day_counts       = cw.groupby('School')['Date'].nunique()
    consistent_schools  = sorted(cw_day_counts[cw_day_counts >= 3].index.tolist())
    consistent_count    = len(consistent_schools)
    pw_day_counts       = pw.groupby('School')['Date'].nunique() if len(pw) else pd.Series(dtype=int)
    prev_consistent     = int((pw_day_counts >= 3).sum())

    # Quiet 7–13 / 30+ days — rolling window anchored to the most recent
    # snapshot date, NOT wall-clock now(). If the operator adds Monday's
    # file but doesn't run the script until Wednesday, "quiet" must still
    # mean "quiet as of Monday's data" — otherwise every school's quiet
    # count inflates by however many days late the run happens to be,
    # even with zero real change in behaviour (this produced a false
    # "quiet 14+" for a school that had actually logged in 13 days before
    # the newest snapshot, purely because the script ran 2 days after it).
    # `pay['School']` is already canonicalised by paying() when the roster
    # is loaded, so name variants of the same school can't double-count.
    today      = pd.Timestamp(combined['Date'].max().date())
    ever       = set(pay['School'].unique())
    last_login = pay.groupby('School')['Date'].max()
    quiet_7  = sorted(s for s in ever if 7 <= (today - last_login[s]).days < 14)
    quiet_30 = sorted(s for s in ever if (today - last_login[s]).days >= 30)

    return {
        'max_week': max_week, 'prev_week': prev_week,
        'cw_logins': cw_logins, 'cw_schools': cw_schools,
        'cw_cls_logins': cw_cls_logins, 'cw_ins_logins': cw_ins_logins,
        'cw_cls_schools': cw_cls_schools, 'cw_ins_schools': cw_ins_schools,
        'pw_logins': pw_logins, 'pw_schools': pw_schools,
        'ever_schools': ever_schools, 'activated_change': activated_change,
        'lifetime_pct': round(ever_schools / TOTAL_CUSTOMERS * 100),
        'weekly_active_pct': round(cw_schools / TOTAL_CUSTOMERS * 100),
        'consistent_schools': consistent_schools, 'consistent_count': consistent_count,
        'prev_consistent': prev_consistent,
        'quiet_7_schools': quiet_7, 'quiet_7_count': len(quiet_7),
        'quiet_30_schools': quiet_30, 'quiet_30_count': len(quiet_30),
    }


def calc_patterns(combined, snap):
    pay       = paying(combined)
    max_week  = snap['max_week']
    prior_set = set(pay[pay['Week'] < max_week]['School'].unique())
    cw_set    = set(pay[pay['Week'] == max_week]['School'].unique())
    return {
        'new_this_week': sorted(cw_set - prior_set),
    }


def calc_weekly_trends(combined):
    pay       = paying(combined)
    all_weeks = sorted(pay['Week'].unique())
    last6     = all_weeks[-6:]

    # Identify which weeks have real product segmentation data
    # (i.e. at least one row with an explicit Product Type column, not defaulted to 'both')
    weeks_with_seg = set()
    if 'Product' in combined.columns:
        # A week has real seg data if it has rows where the product was NOT expanded from 'both'
        # We track this by checking if a week appears in the Feb-23+ file (Week 6+)
        # Simpler: a week has real seg if its cls + ins != 2 × total (both-expansion doubles)
        # Best: check the raw data; since 'both' expands to both products, a week is segmented
        # when cls ≠ total OR ins ≠ total for the deduplicated view
        for w in last6:
            wd  = pay[pay['Week'] == w]
            tot = total_logins(wd)['Email'].count()
            cls = unique_logins(classroom(wd))['Email'].count()
            ins = unique_logins(instrumental(wd))['Email'].count()
            # If cls == tot AND ins == tot, every login was 'both' → no real segmentation
            if tot > 0 and not (cls == tot and ins == tot):
                weeks_with_seg.add(w)

    stats = {}
    for w in last6:
        wd  = pay[pay['Week'] == w]
        tot = total_logins(wd)['Email'].count()
        cls = unique_logins(classroom(wd))['Email'].count()
        ins = unique_logins(instrumental(wd))['Email'].count()
        stats[w] = {
            'schools':     wd['School'].nunique(),
            'logins':      tot,
            'cls':         cls,
            'ins':         ins,
            'segmented':   w in weeks_with_seg,
            'label':       week_label(int(w)),
        }
    return stats, last6


def calc_uk_pilot(combined):
    dm = demo(combined)
    if dm.empty:
        return {'schools': 0, 'logins': 0, 'cls': 0, 'ins': 0,
                'school_list': [], 'has_data': False}

    max_week = int(combined['Week'].max())
    cw = dm[dm['Week'] == max_week]
    return {
        'schools':     cw['School'].nunique(),
        'logins':      total_logins(cw)['Email'].count(),
        'cls':         unique_logins(classroom(cw))['Email'].count(),
        'ins':         unique_logins(instrumental(cw))['Email'].count(),
        'school_list': sorted(cw['School'].unique()),
        'has_data':    not cw.empty,
    }


_TEACHER_ROLES     = {'Teacher', 'School Administrator'}
_PARTICIPANT_ROLES = {'Participant'}


def calc_top10(combined):
    pay = paying(combined)
    max_week = int(pay['Week'].max())

    # Use total_logins (deduplicated by School/Email/Date) per school
    base = total_logins(pay)
    grp  = base.groupby('School').agg(
        total_logins =('Email', 'count'),
        unique_users =('Email', 'nunique'),
        weeks_active =('Week',  'nunique'),
    ).reset_index()

    # Count unique users by role per school using one row per (School, Email)
    unique_users_df = base.drop_duplicates(subset=['School', 'Email'])
    teacher_counts = (
        unique_users_df[unique_users_df['UserRole'].isin(_TEACHER_ROLES)]
        .groupby('School')['Email'].nunique()
        .rename('teacher_count')
    )
    participant_counts = (
        unique_users_df[unique_users_df['UserRole'].isin(_PARTICIPANT_ROLES)]
        .groupby('School')['Email'].nunique()
        .rename('participant_count')
    )
    grp = grp.join(teacher_counts, on='School').join(participant_counts, on='School')
    grp['teacher_count']    = grp['teacher_count'].fillna(0).astype(int)
    grp['participant_count'] = grp['participant_count'].fillna(0).astype(int)

    # Product breakdown per school — REAL per-login counts.
    # Only count rows where the file actually had a Product Type column
    # (ProductExplicit=True). Files from W1–W5 had no Product Type column, so
    # their rows were expanded into both classroom AND instrumental as a
    # fallback; counting those would ghost-count a login in both buckets, so
    # they're excluded here. A genuine "Classroom & Instrumental" user legitimately
    # counts in both (that's a real dual-product login), which unique_logins handles.
    #
    # This replaces the old logic that dumped a school's FULL total into BOTH
    # cls and ins whenever it was tagged "both" — which made an essentially
    # instrumental-only school (e.g. Koa: 4 classroom vs 698 instrumental logins)
    # display a wildly wrong "875 cls / 875 ins".
    explicit_pay = pay[pay['ProductExplicit'] == True]
    cls_by_school = (unique_logins(classroom(explicit_pay))
                     .groupby('School')['Email'].count())
    ins_by_school = (unique_logins(instrumental(explicit_pay))
                     .groupby('School')['Email'].count())

    grp['cls'] = grp['School'].map(cls_by_school).fillna(0).astype(int)
    grp['ins'] = grp['School'].map(ins_by_school).fillna(0).astype(int)

    grp['score'] = grp['weeks_active'] * grp['unique_users'] * 2 + grp['total_logins']
    grp = grp.sort_values('score', ascending=False).head(10).reset_index(drop=True)

    result = []
    for _, row in grp.iterrows():
        school_weeks = sorted(pay[pay['School'] == row['School']]['Week'].unique())
        weeks_str    = ', '.join(f"W{int(w)}" for w in school_weeks)
        # Derive the display label from the real counts: both non-zero → 'both',
        # one non-zero → that product, neither → None (no product data at all).
        cls_n, ins_n = int(row['cls']), int(row['ins'])
        if cls_n and ins_n:
            known_product = 'both'
        elif cls_n:
            known_product = 'classroom'
        elif ins_n:
            known_product = 'instrumental'
        else:
            known_product = None
        result.append({
            'name':              row['School'],
            'logins':            int(row['total_logins']),
            'teacher_count':     int(row['teacher_count']),
            'participant_count': int(row['participant_count']),
            'known_product':     known_product,
            'cls':               cls_n,
            'ins':               ins_n,
            'weeks':             weeks_str,
            'weeks_active':      int(row['weeks_active']),
            'in_latest':         max_week in [int(w) for w in school_weeks],
        })
    return result


# ── HTML builders ─────────────────────────────────────────────────────────────

def build_daily_pulse_html(dp):
    if not dp:
        return '<section class="dashboard-section"><p>No data.</p></section>'

    yesterday_str  = strf(dp['yesterday'], '%A, %-d %B %Y')
    day_before_str = strf(dp['day_before'], '%A, %-d %B') if dp['day_before'] else '–'

    login_delta  = pct_change_html(dp['y_logins'],  dp['db_logins'])
    school_delta = pct_change_html(dp['y_schools'], dp['db_schools'])

    new_html = ''
    if dp['new_schools']:
        badges = ''.join(f'<span class="school-badge new-badge">{s}</span>'
                         for s in dp['new_schools'])
        new_html = f'<div class="pulse-new-schools"><div class="pulse-new-label">🆕 New today vs yesterday</div><div class="badge-row">{badges}</div></div>'
    else:
        new_html = '<div class="pulse-new-schools"><div class="pulse-new-label">No new schools vs yesterday</div></div>'

    return f'''
        <!-- ═══════════════════════════════════ DAILY PULSE ═══════════════════════════════════ -->
        <section class="dashboard-section" id="section-daily-pulse">
            <h2 class="section-title turkish">⚡ Daily Pulse</h2>
            <p class="section-desc">Yesterday · paying customers only</p>

            <div class="pulse-grid">
                <div class="pulse-card">
                    <div class="pulse-date">{yesterday_str}</div>
                    <div class="pulse-row">
                        <div class="pulse-metric">
                            <div class="pulse-value" id="pulse-logins">{dp['y_logins']}</div>
                            <div class="pulse-label">Logins {login_delta}</div>
                            <div class="pulse-split">
                                <span class="split-cls">🏫 {dp['y_cls']} classroom</span>
                                <span class="split-sep">·</span>
                                <span class="split-ins">🎵 {dp['y_ins']} instrumental</span>
                            </div>
                        </div>
                        <div class="pulse-divider"></div>
                        <div class="pulse-metric">
                            <div class="pulse-value" id="pulse-schools">{dp['y_schools']}</div>
                            <div class="pulse-label">Schools {school_delta}</div>
                            <div class="pulse-split">paying customers</div>
                        </div>
                    </div>
                    <div class="pulse-prev">vs {day_before_str}: {dp['db_logins']} logins · {dp['db_schools']} schools</div>
                    {new_html}
                </div>
            </div>
        </section>'''


def build_weekly_snapshot_html(snap):
    mw = snap['max_week']
    pw = snap['prev_week']

    logins_delta  = pct_change_html(snap['cw_logins'],  snap['pw_logins'])
    schools_delta = pct_change_html(snap['cw_schools'], snap['pw_schools'])
    act_change    = snap['activated_change']
    act_delta     = (f'<span class="delta up">↗ +{act_change} new activations</span>' if act_change > 0
                     else '<span class="delta flat">→ no new activations</span>')
    cons_delta    = pct_change_html(snap['consistent_count'], snap['prev_consistent'])

    cons_badges = ''.join(f'<span class="school-badge cons-badge">{s}</span>'
                          for s in snap['consistent_schools']) or '<em style="color:var(--gray)">None yet</em>'
    quiet_7_badges = ''.join(f'<span class="school-badge quiet-badge">{s}</span>'
                             for s in snap['quiet_7_schools']) or '<em style="color:var(--gray)">All schools active this week!</em>'
    quiet_badges = ''.join(f'<span class="school-badge quiet-badge">{s}</span>'
                           for s in snap['quiet_30_schools']) or '<em style="color:var(--gray)">All schools active!</em>'

    return f'''
        <!-- ═══════════════════════════════ WEEKLY SNAPSHOT ══════════════════════════════════ -->
        <section class="dashboard-section" id="section-weekly-snapshot">
            <h2 class="section-title pacific">📋 Weekly Snapshot — Week {mw}</h2>
            <p class="section-desc">Paying customers only · Week {mw} vs Week {pw}</p>

            <!-- Row 1: core metrics -->
            <div class="snapshot-grid-top">

                <!-- Card 1: This week vs Last week -->
                <div class="snap-card accent-pacific">
                    <div class="snap-label">Week {mw} vs Week {pw}</div>
                    <div class="snap-body">
                        <div class="snap-segment-row">
                            <span class="seg-icon">🏫</span>
                            <span class="seg-label">Classroom</span>
                            <span class="seg-value">{snap['cw_cls_logins']}</span>
                            <span class="seg-sub">logins · {snap['cw_cls_schools']} schools</span>
                        </div>
                        <div class="snap-segment-row">
                            <span class="seg-icon">🎵</span>
                            <span class="seg-label">Instrumental</span>
                            <span class="seg-value">{snap['cw_ins_logins']}</span>
                            <span class="seg-sub">logins · {snap['cw_ins_schools']} schools</span>
                        </div>
                        <div class="snap-total-row">
                            <span class="snap-value" id="snap-cw-logins">{snap['cw_logins']}</span>
                            <span class="snap-unit">total logins {logins_delta}</span>
                            <span class="snap-unit">{snap['cw_schools']} schools {schools_delta}</span>
                        </div>
                    </div>
                    <div class="snap-prev">Last week: {snap['pw_logins']} logins · {snap['pw_schools']} schools</div>
                </div>

                <!-- Card 2: Schools Activated -->
                <div class="snap-card accent-forest">
                    <div class="snap-label">Schools Activated</div>
                    <div class="snap-body">
                        <div class="snap-segment-row">
                            <span class="seg-label">Lifetime</span>
                            <span class="seg-value" id="snap-activated">{snap['ever_schools']}</span>
                            <span class="seg-sub">of {TOTAL_CUSTOMERS} ({snap['lifetime_pct']}%)</span>
                        </div>
                        <div class="snap-segment-row">
                            <span class="seg-label">This week</span>
                            <span class="seg-value">{snap['cw_schools']}</span>
                            <span class="seg-sub">of {TOTAL_CUSTOMERS} ({snap['weekly_active_pct']}%)</span>
                        </div>
                    </div>
                    <div class="snap-prev">{act_delta}</div>
                </div>

                <!-- Card 3: Consistent Users -->
                <div class="snap-card accent-lilac">
                    <div class="snap-label">Consistent This Week</div>
                    <div class="snap-body">
                        <div class="snap-metric">
                            <span class="snap-value" id="snap-consistent">{snap['consistent_count']}</span>
                            <span class="snap-unit">schools 3+ days {cons_delta}</span>
                        </div>
                    </div>
                    <div class="snap-badges">{cons_badges}</div>
                </div>

            </div>

            <!-- Row 2: quiet cards — always side-by-side with scrollable school lists -->
            <div class="snapshot-grid-quiet">

                <!-- Card 4: Quiet 7-13 Days -->
                <div class="snap-card accent-salmon">
                    <div class="snap-label">Quiet 7–13 Days</div>
                    <div class="snap-body">
                        <div class="snap-metric">
                            <span class="snap-value" id="snap-quiet-7">{snap['quiet_7_count']}</span>
                            <span class="snap-unit">schools · last login 1–2 weeks ago</span>
                        </div>
                    </div>
                    <div class="snap-badges snap-badges-scroll">{quiet_7_badges}</div>
                </div>

                <!-- Card 5: Quiet 30+ Days -->
                <div class="snap-card accent-salmon">
                    <div class="snap-label">Quiet 30+ Days</div>
                    <div class="snap-body">
                        <div class="snap-metric">
                            <span class="snap-value" id="snap-quiet">{snap['quiet_30_count']}</span>
                            <span class="snap-unit">schools · last login 30+ days ago</span>
                        </div>
                    </div>
                    <div class="snap-badges snap-badges-scroll">{quiet_badges}</div>
                </div>

            </div>
        </section>'''


def build_uk_pilot_html(uk):
    if not uk['has_data']:
        schools_html = '<div class="pilot-empty">No Demo/Pilot schools have logged in yet.</div>'
    else:
        badges = ''.join(f'<span class="school-badge new-badge">{s}</span>'
                         for s in uk['school_list'])
        schools_html = f'<div class="badge-row">{badges}</div>'

    return f'''
        <!-- ═══════════════════════════════ UK PILOT ═══════════════════════════════════════ -->
        <section class="dashboard-section" id="section-uk-pilot">
            <h2 class="section-title forest">🇬🇧 UK Pilot</h2>
            <p class="section-desc">Demo schools · not counted in main metrics</p>

            <div class="pilot-grid">
                <div class="pilot-stat">
                    <div class="pilot-value">{uk['schools']}</div>
                    <div class="pilot-label">Schools active this week</div>
                </div>
                <div class="pilot-stat">
                    <div class="pilot-value">{uk['logins']}</div>
                    <div class="pilot-label">Logins this week</div>
                </div>
                <div class="pilot-stat">
                    <div class="pilot-value">{uk['cls']}</div>
                    <div class="pilot-label">🏫 Classroom</div>
                </div>
                <div class="pilot-stat">
                    <div class="pilot-value">{uk['ins']}</div>
                    <div class="pilot-label">🎵 Instrumental</div>
                </div>
            </div>
            {schools_html}
        </section>'''


def build_patterns_html(patterns, snap):
    mw = snap['max_week']

    if patterns['new_this_week']:
        new_badges = ''.join(f'<span class="school-badge new-badge">{s}</span>'
                             for s in patterns['new_this_week'])
        new_html = f'<div class="pattern-count-badge">{len(patterns["new_this_week"])}</div><div class="badge-row">{new_badges}</div>'
    else:
        new_html = '<div class="pattern-empty">No first-time logins this week.</div>'

    return f'''
        <!-- ═══════════════════════════════ PATTERNS THIS WEEK ═══════════════════════════════ -->
        <section class="dashboard-section" id="section-patterns">
            <h2 class="section-title forest">🔍 Patterns This Week</h2>
            <p class="section-desc">Paying customers only · Week {mw}</p>

            <div class="patterns-grid patterns-grid-2col">
                <div class="pattern-block">
                    <div class="pattern-block-title">🆕 New Activations</div>
                    <div class="pattern-block-desc">First-ever login this week (paying)</div>
                    {new_html}
                </div>
                <div class="pattern-block pattern-notes">
                    <div class="pattern-block-title">📝 Notes</div>
                    <div class="pattern-block-desc">Contextual insights for this week</div>
                    <textarea class="notes-field" id="weekly-notes" placeholder="Add your observations here…&#10;e.g. Koa Academy instrumental signups spiked&#10;Reached out to School X re: inactivity…"></textarea>
                </div>
            </div>
        </section>'''


def build_trends_html(weekly_stats, last6):
    if not last6:
        return ''

    latest_week = max(last6)
    first_week  = min(last6)
    ls = weekly_stats[latest_week]['schools']
    fs = weekly_stats[first_week]['schools']

    if fs > 0:
        gpct = round(((ls - fs) / fs) * 100)
        growth_text = (f'↗ +{gpct}% vs {weekly_stats[first_week]["label"]}' if gpct > 0
                       else f'↘ {gpct}% vs {weekly_stats[first_week]["label"]}' if gpct < 0
                       else '→ Steady')
    else:
        growth_text = '📊 First data point'

    # Max total logins (for bar scaling)
    max_logins = max(weekly_stats[w]['logins'] for w in last6) or 1

    # Legend + bars
    bars_html = ''
    for w in reversed(last6):
        ws        = weekly_stats[w]
        total     = ws['logins']
        cls       = ws['cls']
        ins       = ws['ins']
        segmented = ws['segmented']
        is_lat    = ' bar-latest' if w == latest_week else ''
        label     = ws['label']
        tot_pct   = max(int((total / max_logins) * 100), 2) if total else 0

        if segmented and total > 0:
            # Real product data → stacked bar
            cls_pct = int((cls / total) * tot_pct)
            ins_pct = tot_pct - cls_pct
            stats_text = f'{ws["schools"]} schools · {total} logins ({cls} cls / {ins} ins)'
            bar_inner = (f'<div class="bar-segment bar-cls" style="width:{cls_pct}%"></div>'
                         f'<div class="bar-segment bar-ins" style="width:{ins_pct}%"></div>'
                         f'<span class="bar-total-label">{total}</span>')
        else:
            # Old data with no product split → single unified bar
            cls_pct = tot_pct
            stats_text = f'{ws["schools"]} schools · {total} logins'
            bar_inner = (f'<div class="bar-segment bar-cls" style="width:{cls_pct}%"></div>'
                         f'<span class="bar-total-label">{total}</span>')

        bars_html += f'''
                <div class="week-bar-item">
                    <div class="week-info">
                        <span class="week-label">{label}</span>
                        <span class="week-stats">{stats_text}</span>
                    </div>
                    <div class="bar-container">
                        <div class="stacked-bar{is_lat}">
                            {bar_inner}
                        </div>
                    </div>
                </div>'''

    legend_html = '''
            <div class="bar-legend">
                <span class="legend-item"><span class="legend-swatch swatch-cls"></span> Classroom</span>
                <span class="legend-item"><span class="legend-swatch swatch-ins"></span> Instrumental</span>
                <span class="legend-item" style="color:#aaa;font-style:italic;">Single-colour = no product data yet</span>
            </div>'''

    return f'''
        <!-- ════════════════════════════════ WEEKLY TRENDS ═══════════════════════════════════ -->
        <section class="dashboard-section" id="section-trends">
            <h2 class="section-title pacific">📈 Weekly Trends</h2>
            <p class="section-desc">Last {len(last6)} weeks · paying customers · classroom vs instrumental</p>

            <div class="trend-summary">
                <div class="trend-label">Latest Week</div>
                <div class="trend-value" id="thisWeekSchools">{ls} schools</div>
                <div class="trend-change" id="weekGrowth">{growth_text}</div>
            </div>

            {legend_html}
            <div class="weeks-container" id="weeks-container">
{bars_html}
            </div>
        </section>'''


def _user_count_str(teacher_count, participant_count):
    """Return e.g. '5 teachers • 12 students', '5 teachers', or '12 students'."""
    parts = []
    if teacher_count:
        parts.append(f"{teacher_count} {'teacher' if teacher_count == 1 else 'teachers'}")
    if participant_count:
        parts.append(f"{participant_count} {'student' if participant_count == 1 else 'students'}")
    return ' • '.join(parts) if parts else ''


def _product_str(known_product, cls, ins):
    """
    Build the cls/ins display fragment.
    Only shown when we have real Product Type data (known_product is not None).
      instrumental          → '🎵 230 ins'
      classroom             → '🏫 50 cls'
      both                  → '🏫 X cls / 🎵 Y ins'
      None (no data)        → ''   (omitted entirely from the stats line)
    """
    if known_product is None:
        return ''
    if known_product == 'instrumental':
        return f'🎵 {ins} ins'
    if known_product == 'classroom':
        return f'🏫 {cls} cls'
    return f'🏫 {cls} cls / 🎵 {ins} ins'   # 'both'


def build_top10_html(top10):
    items_html = ''
    for i, s in enumerate(top10):
        badge_class = ('badge-core'   if s['weeks_active'] >= 4
                       else 'badge-active' if s['in_latest']
                       else 'badge-quiet')
        badge_text  = ('🏆 Core'   if s['weeks_active'] >= 4
                       else '✅ Active' if s['in_latest']
                       else '💤 Quiet')
        user_str    = _user_count_str(s['teacher_count'], s['participant_count'])
        user_part   = f' · {user_str}' if user_str else ''
        product_str = _product_str(s['known_product'], s['cls'], s['ins'])
        product_part = f' · {product_str}' if product_str else ''
        items_html += f'''
                <li class="top-school-item">
                    <div class="rank-number">{i+1}</div>
                    <div class="school-info">
                        <div class="school-name">{s['name']}</div>
                        <div class="school-stats">{s['logins']} logins{user_part} · {s['weeks']}{product_part}</div>
                    </div>
                    <span class="pattern-badge {badge_class}">{badge_text}</span>
                </li>'''

    return f'''
        <!-- ═══════════════════════════════════ TOP 10 ═══════════════════════════════════════ -->
        <section class="dashboard-section" id="section-top10">
            <h2 class="section-title turkish">🏆 Top 10 Schools</h2>
            <p class="section-desc">Paying customers · ranked by consistency × teachers × frequency</p>
            <ul class="top-schools-list">{items_html}
            </ul>
        </section>'''


def calc_lifetime_logins(combined):
    """Total logins per paying school across all weeks, sorted descending.

    Roster mode: returns ALL roster schools, including silent ones with 0
    logins, so the dashboard reflects the full AC paying-customer list.
    Display names prefer the most-frequent snapshot name observed (operator-
    familiar); silent schools fall back to a cleaned roster title.
    """
    pay  = paying(combined)
    base = total_logins(pay)

    if _ROSTER_LOOKUP is None or not _ROSTER:
        # Legacy: snapshot-only path
        grp = (base.groupby('School')['Email']
                    .count()
                    .reset_index()
                    .rename(columns={'Email': 'total_logins'}))
        grp = grp.sort_values('total_logins', ascending=False).reset_index(drop=True)
        return grp.to_dict('records')

    # Roster mode: tag every snapshot row with its roster deal_id, then group.
    base = base.copy()
    base['roster_deal_id'] = base['School'].apply(
        lambda n: (resolve_to_roster(n) or {}).get('deal_id'))

    counted = (base.dropna(subset=['roster_deal_id'])
                    .groupby('roster_deal_id')['Email']
                    .count()
                    .to_dict())

    # Best display name per deal_id = the most-common snapshot name we saw.
    name_counter = {}
    for sname, did in base[['School', 'roster_deal_id']].dropna().itertuples(index=False):
        name_counter.setdefault(did, Counter())[sname] += 1

    records = []
    for entry in _ROSTER:
        did     = entry['deal_id']
        n       = int(counted.get(did, 0))
        display = (name_counter[did].most_common(1)[0][0]
                   if did in name_counter and name_counter[did]
                   else clean_roster_display_name(entry))
        records.append({
            'School':       display,
            'total_logins': n,
            'stage':        entry.get('stage', ''),
            'silent':       n == 0,
        })

    # Sort: active schools first by logins desc, silent at the bottom alphabetically.
    records.sort(key=lambda r: (r['silent'], -r['total_logins'], r['School'].lower()))
    return records


CAP_SCHOOL = 'Koa Academy'
CAP_AT     = 200   # display cap for bar width only — actual count still shown


def build_lifetime_logins_html(lifetime_data):
    if not lifetime_data:
        return ''

    # All bars scale against CAP_AT so the capped school sits at 100 %
    # and every other school is proportional to that reference.
    scale_max = CAP_AT

    silent_count = sum(1 for s in lifetime_data if s.get('silent') or s['total_logins'] == 0)
    active_count = len(lifetime_data) - silent_count

    bars_html = ''
    for s in lifetime_data:
        actual    = s['total_logins']
        is_silent = bool(s.get('silent')) or actual == 0
        is_capped = (s['School'] == CAP_SCHOOL and actual > CAP_AT)
        bar_val   = CAP_AT if is_capped else actual
        pct       = min(max(int((bar_val / scale_max) * 100), 2), 100) if actual else 0

        if is_silent:
            label    = '<em class="lifetime-silent-label">Not yet active</em>'
            cap_attr = ''
            fill_cls = 'lifetime-bar-fill lifetime-bar-silent'
        else:
            label    = f'{actual} ↗' if is_capped else str(actual)
            cap_attr = (' title="Bar capped at 200 for readability — actual total shown"'
                        if is_capped else '')
            fill_cls = 'lifetime-bar-fill lifetime-bar-capped' if is_capped else 'lifetime-bar-fill'

        bars_html += f'''
                <div class="lifetime-bar-item{' lifetime-bar-item-silent' if is_silent else ''}">
                    <div class="lifetime-school-name">{s['School']}</div>
                    <div class="lifetime-bar-wrap">
                        <div class="lifetime-bar-track"{cap_attr}>
                            <div class="{fill_cls}" style="width:{pct}%"></div>
                        </div>
                        <span class="lifetime-bar-value">{label}</span>
                    </div>
                </div>'''

    school_count = len(lifetime_data)
    return f'''
        <!-- ════════════════════ ALL SCHOOLS LIFETIME LOGINS ════════════════════════ -->
        <section class="dashboard-section" id="section-lifetime">
            <h2 class="section-title pacific">📊 All Paying Schools — Lifetime Activity</h2>
            <p class="section-desc">All {school_count} AC paying schools · {active_count} active · {silent_count} not yet active · sorted by total logins</p>

            <div class="lifetime-grid">
{bars_html}
            </div>
        </section>'''


# ── Usage Patterns tab ───────────────────────────────────────────────────────
# Classifies every AC-roster school into a usage pattern based on snapshot
# activity. Rules are intentionally simple and reproducible — tune the
# constants below if the categories need adjustment.

PATTERN_RULES = {
    'power_min_tl':         100,    # Power User: ≥100 lifetime logins
    'power_min_users':      50,     # Power User: ≥50 distinct users (teachers + students)
    'highfreq_min_tl':      20,     # High Frequency: ≥20 lifetime logins
    'highfreq_min_per_wk':  4.5,    # High Frequency: ≥4.5 logins per active week
    'consistent_min_ratio': 0.6,    # Consistent: active in ≥60% of weeks since first login
    'consistent_lowvol_max_per_wk': 2,  # Low-Vol Consistent: ≤2 logins per active week
    'early_stage_max_uw':   3,      # Early Stage: <4 active weeks
    'quiet_min_weeks':      6,      # Gone Quiet overlay: ≥6 weeks since last login
}

PATTERN_DESCRIPTIONS = {
    'Power User':            'Extremely high login volume with many active users (teachers and students). These schools have deeply embedded Ear Academy into their program — worth understanding what they do differently and using as case studies.',
    'High Frequency':        'Multiple logins per active week. These schools use the platform intensively in bursts. Watch for big spikes followed by silence — worth a gentle check-in.',
    'Consistent Weekly':     'Present in 60%+ of weeks since joining. Reliable, habitual usage — your most stable accounts. Low churn risk.',
    'Consistent Low-Volume': '1–2 logins per active week, often a single teacher. Do not confuse low volume with low commitment — quietly but reliably engaged.',
    'Bi-weekly':             'Active roughly every other week, likely matching a fortnightly class schedule. Natural pattern, not a warning sign.',
    'Early Stage':           'Fewer than 4 active weeks. Too early to classify — monitor over the next few weeks.',
    'One-time':              'Only appeared in one snapshot. No pattern established yet.',
    'Not Yet Active':        'A paying school per ActiveCampaign, but no snapshot logins yet. Onboarding follow-up may be needed.',
    'quiet':                 'No login activity for 6+ weeks. Some may have natural seasonal gaps — check their history before reaching out.',
}

# Display colours per pattern (used by the heatmap legend / badge styles)
PATTERN_COLORS = {
    'Power User':            {'bg': '#0F6E5618', 'text': '#085041', 'border': '#0F6E56'},
    'High Frequency':        {'bg': '#00a19a18', 'text': '#006b66', 'border': '#00a19a'},
    'Consistent Weekly':     {'bg': '#1d70b818', 'text': '#0C447C', 'border': '#1d70b8'},
    'Consistent Low-Volume': {'bg': '#534AB718', 'text': '#3C3489', 'border': '#534AB7'},
    'Bi-weekly':             {'bg': '#38aae118', 'text': '#1a6a9a', 'border': '#38aae1'},
    'Early Stage':           {'bg': '#d8d3cb40', 'text': '#666',    'border': '#d8d3cb'},
    'One-time':              {'bg': '#E8E4DF50', 'text': '#888',    'border': '#d8d3cb'},
    'Not Yet Active':        {'bg': '#F8E7D5',   'text': '#8a4a1f', 'border': '#d49a55'},
}

# Display order for the filter pills (and matches the heatmap sort priority)
PATTERN_ORDER = [
    'Power User', 'High Frequency', 'Consistent Weekly',
    'Consistent Low-Volume', 'Bi-weekly', 'Early Stage',
    'One-time', 'Not Yet Active',
]


def _classify_pattern(tl, uw, ut, weeks_span, weeks_since_last):
    """Return (pattern_name, is_quiet) for a single school."""
    r = PATTERN_RULES
    if tl == 0:
        return 'Not Yet Active', False
    quiet = weeks_since_last >= r['quiet_min_weeks']

    if tl >= r['power_min_tl'] and ut >= r['power_min_users']:
        return 'Power User', quiet
    if uw == 1:
        return 'One-time', quiet
    if uw <= r['early_stage_max_uw']:
        return 'Early Stage', quiet
    per_week = tl / max(uw, 1)
    if tl >= r['highfreq_min_tl'] and per_week >= r['highfreq_min_per_wk']:
        return 'High Frequency', quiet
    consistency = uw / max(weeks_span, 1)
    if consistency >= r['consistent_min_ratio']:
        if per_week <= r['consistent_lowvol_max_per_wk']:
            return 'Consistent Low-Volume', quiet
        return 'Consistent Weekly', quiet
    return 'Bi-weekly', quiet


def calc_usage_patterns(combined):
    """Build the data feeding the Usage Patterns tab.

    Output schema:
      {
        'weeks': [iso-date strings, one per snapshot week, ascending],
        'schools': [ {'s', 'tl', 'uw', 'ut', 'p', 'q', 'd': {iso: count}} ],
        'pattern_counts': { pattern_name: count },
        'totals': {
            'schools_tracked': int,    # = len(roster) when roster present
            'total_logins': int,
            'weeks_of_data': int,
            'gone_quiet': int,
            'consistent_users': int,   # Consistent Weekly + Low-Volume
            'bi_weekly_users': int,
            'not_yet_active': int,
        },
        'date_range_label': 'Jan 19 – May 12, 2026',
      }
    """
    pay = paying(combined)
    if pay.empty:
        return None

    # Determine the snapshot week-anchor for each row: Monday of the snapshot date.
    pay = pay.copy()
    pay['WeekStart'] = (pay['Date'] - pd.to_timedelta(pay['Date'].dt.weekday, unit='D')).dt.normalize()

    weeks_sorted = sorted(pay['WeekStart'].unique())
    weeks_iso    = [pd.Timestamp(w).strftime('%Y-%m-%d') for w in weeks_sorted]
    latest_week  = pd.Timestamp(weeks_sorted[-1])

    # Resolve each row to its roster entry so the school list matches the roster.
    if _ROSTER_LOOKUP is not None:
        pay['DealId'] = pay['School'].apply(
            lambda n: (resolve_to_roster(n) or {}).get('deal_id'))
        pay = pay.dropna(subset=['DealId'])
    else:
        pay['DealId'] = pay['School']  # legacy fallback: use snapshot name as key

    # Pick the most-common snapshot name per deal_id as the display name.
    name_counter = {}
    for sname, did in pay[['School', 'DealId']].itertuples(index=False):
        name_counter.setdefault(did, Counter())[sname] += 1
    display_name_for = {did: apply_display_override(ctr.most_common(1)[0][0])
                        for did, ctr in name_counter.items()}

    # Per-school per-week login counts (deduplicated by School+Email+Date).
    base = total_logins(pay)
    base = base[['DealId', 'WeekStart', 'Email', 'UserRole', 'Date']]
    per_school_week = (base.groupby(['DealId', 'WeekStart'])['Email']
                            .count()
                            .reset_index()
                            .rename(columns={'Email': 'logins'}))

    schools_out = []
    # First, every active roster school (or every snapshot deal_id in legacy mode)
    active_ids = set(per_school_week['DealId'].unique())
    for did in active_ids:
        rows = per_school_week[per_school_week['DealId'] == did]
        d_map = {pd.Timestamp(w).strftime('%Y-%m-%d'): int(n)
                 for w, n in zip(rows['WeekStart'], rows['logins'])}
        tl = sum(d_map.values())
        uw = len(d_map)

        sub = base[base['DealId'] == did]
        ut = sub['Email'].nunique()                                   # total unique users
        # One role per user (first seen), matching the Top 10 tab's method, so
        # a user who appears under different roles on different dates isn't
        # double-counted as both a teacher and a student.
        sub_users = sub.drop_duplicates(subset=['Email'])
        tc = sub_users[sub_users['UserRole'].isin(_TEACHER_ROLES)]['Email'].nunique()      # teachers/admins
        sc = sub_users[sub_users['UserRole'].isin(_PARTICIPANT_ROLES)]['Email'].nunique()  # students

        first_week = pd.Timestamp(sub['WeekStart'].min())
        weeks_span = max(int((latest_week - first_week).days // 7) + 1, 1)

        last_week  = pd.Timestamp(sub['WeekStart'].max())
        weeks_since_last = int((latest_week - last_week).days // 7)

        # NOTE: Power User etc. use `ut` (total unique users), not teachers.
        pattern, quiet = _classify_pattern(tl, uw, ut, weeks_span, weeks_since_last)
        schools_out.append({
            's':  display_name_for.get(did, str(did)),
            'tl': tl, 'uw': uw, 'ut': ut, 'tc': int(tc), 'sc': int(sc),
            'p':  pattern, 'q': quiet, 'd': d_map,
        })

    # Then, silent roster schools — they get 'Not Yet Active' and empty d_map.
    if _ROSTER:
        for entry in _ROSTER:
            if entry['deal_id'] in active_ids:
                continue
            schools_out.append({
                's':  apply_display_override(clean_roster_display_name(entry)),
                'tl': 0, 'uw': 0, 'ut': 0, 'tc': 0, 'sc': 0,
                'p':  'Not Yet Active', 'q': False,
                'd':  {w: 0 for w in weeks_iso},
            })

    schools_out.sort(key=lambda x: (-x['tl'], x['s'].lower()))

    pattern_counts = Counter(s['p'] for s in schools_out)
    gone_quiet     = sum(1 for s in schools_out if s['q'])
    consistent     = pattern_counts.get('Consistent Weekly', 0) + pattern_counts.get('Consistent Low-Volume', 0)
    bi_weekly      = pattern_counts.get('Bi-weekly', 0)
    not_yet_active = pattern_counts.get('Not Yet Active', 0)

    schools_tracked = len(_ROSTER) if _ROSTER else len(schools_out)
    total_logins_v  = sum(s['tl'] for s in schools_out)

    # Label the range by the ACTUAL data span, not the week-start Monday of the
    # last bucket. Using weeks_sorted[-1] (a Monday) made this tab read
    # "… – 17 Aug" while the rest of the dashboard already showed 20 Aug —
    # looking 3 days stale even though the latest logins are present (they're
    # just aggregated into the current week's heatmap column). Anchor to
    # combined's real min/max so it matches the "Updated" date everywhere else.
    first_str = strf(pd.Timestamp(combined['Date'].min()), '%-d %b')
    last_str  = strf(pd.Timestamp(combined['Date'].max()), '%-d %b %Y')
    date_range_label = f'{first_str} – {last_str}'

    return {
        'weeks':        weeks_iso,
        'schools':      schools_out,
        'pattern_counts': dict(pattern_counts),
        'totals': {
            'schools_tracked': schools_tracked,
            'total_logins':    total_logins_v,
            'weeks_of_data':   len(weeks_iso),
            'gone_quiet':      gone_quiet,
            'consistent_users': consistent,
            'bi_weekly_users':  bi_weekly,
            'not_yet_active':   not_yet_active,
        },
        'date_range_label': date_range_label,
    }


def build_usage_patterns_html(p):
    """Build the visible HTML block (summary tiles + filter pills) for the tab.
    Returns the HTML string to drop between PATTERNS_BLOCK_START/END markers.
    """
    if not p:
        return '<p>No usage-pattern data available.</p>'

    t   = p['totals']
    pc  = p['pattern_counts']
    n_schools = t['schools_tracked']
    n_snapshots = sum(1 for _ in p['weeks'])

    def pill(pattern_key, label, color_text):
        count = pc.get(pattern_key, 0)
        return (f'<button class="pt-filter-btn" data-pattern="{pattern_key}" '
                f'style="color:{color_text}">{label} ({count})</button>')

    pills_html = (
        f'<button class="pt-filter-btn active" data-pattern="all" '
        f'style="color:var(--dark)">All schools ({n_schools})</button>'
        + pill('Power User',            '⚡ Power user',          '#085041')
        + pill('High Frequency',        '🔥 High frequency',      'var(--green)')
        + pill('Consistent Weekly',     '📅 Consistent weekly',   'var(--lapis)')
        + pill('Consistent Low-Volume', '📆 Low-vol consistent',  '#534AB7')
        + pill('Bi-weekly',             '〰 Bi-weekly',           'var(--sky)')
        + pill('Early Stage',           '🌱 Early stage',         'var(--gray)')
        + pill('One-time',              '👋 One-time',            '#B4B2A9')
        + pill('Not Yet Active',        '🆕 Not yet active',      '#8a4a1f')
        + f'<button class="pt-filter-btn" data-pattern="quiet" '
          f'style="color:#b91c1c">🔴 Gone quiet ({t["gone_quiet"]})</button>'
    )

    return f'''<h2 class="section-title pacific">🔍 Usage Patterns — {p["date_range_label"]}</h2>
        <p class="section-desc">{n_snapshots} weekly snapshots · {n_schools} paying schools · {p["date_range_label"]} · Click a filter to explore</p>

        <div class="pt-summary-grid">
            <div class="pt-summary-card"><div class="pt-card-top" style="background:var(--green)"></div><div class="pt-card-num">{n_schools}</div><div class="pt-card-label">Paying schools (AC)</div></div>
            <div class="pt-summary-card"><div class="pt-card-top" style="background:var(--sky)"></div><div class="pt-card-num">{t["total_logins"]}</div><div class="pt-card-label">Total logins</div></div>
            <div class="pt-summary-card"><div class="pt-card-top" style="background:var(--lapis)"></div><div class="pt-card-num">{t["weeks_of_data"]}</div><div class="pt-card-label">Weeks of data</div></div>
            <div class="pt-summary-card"><div class="pt-card-top" style="background:#b91c1c"></div><div class="pt-card-num" style="color:#b91c1c">{t["gone_quiet"]}</div><div class="pt-card-label">Gone quiet 6+ wks</div></div>
            <div class="pt-summary-card"><div class="pt-card-top" style="background:var(--gold2)"></div><div class="pt-card-num">{t["consistent_users"]}</div><div class="pt-card-label">Consistent users</div></div>
            <div class="pt-summary-card"><div class="pt-card-top" style="background:var(--shale)"></div><div class="pt-card-num">{t["bi_weekly_users"]}</div><div class="pt-card-label">Bi-weekly users</div></div>
            <div class="pt-summary-card"><div class="pt-card-top" style="background:#d49a55"></div><div class="pt-card-num" style="color:#8a4a1f">{t["not_yet_active"]}</div><div class="pt-card-label">Not yet active</div></div>
        </div>

        <div class="pt-filters" id="pt-filters">
            {pills_html}
        </div>

        <div class="pt-desc-box" id="pt-desc-box"></div>

        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;flex-wrap:wrap;gap:0.5rem;">
            <div style="font-size:0.8rem;color:var(--gray);">Showing <strong id="pt-count">{n_schools}</strong> schools &nbsp;·&nbsp; Heatmap: {p["date_range_label"]}</div>
            <div style="display:flex;gap:12px;font-size:0.75rem;color:var(--gray);align-items:center;">
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#0F6E56;vertical-align:middle;margin-right:3px;"></span>High</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#5DCAA5;vertical-align:middle;margin-right:3px;"></span>Med</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#c8eed9;vertical-align:middle;margin-right:3px;"></span>Low</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#E8E4DF;vertical-align:middle;margin-right:3px;"></span>None</span>
            </div>
        </div>

        <div class="pt-table-header">
            <div>School</div>
            <div id="pt-week-headers" style="display:flex;gap:3px;"></div>
            <div>Pattern</div>
        </div>
        <div class="pt-school-list" id="pt-school-list"></div>'''


def build_usage_patterns_js(p):
    """Build the JS data block (WEEKS, SCHOOLS, PC, DESC) for the heatmap.
    Returns the JS source to drop between PATTERNS_DATA_START/END markers.
    """
    if not p:
        return 'var WEEKS=[];var SCHOOLS=[];var PC={};var DESC={};'

    weeks_js   = json.dumps(p['weeks'])
    schools_js = json.dumps(p['schools'], ensure_ascii=False)
    pc_js      = json.dumps(PATTERN_COLORS, ensure_ascii=False)
    desc_js    = json.dumps(PATTERN_DESCRIPTIONS, ensure_ascii=False)

    return (f'var WEEKS={weeks_js};\n'
            f'var SCHOOLS={schools_js};\n'
            f'var PC={pc_js};\n'
            f'var DESC={desc_js};')


# ── UK Pilots tab ────────────────────────────────────────────────────────────
# UK pilot schools are NOT in the AC paying roster, so they're dropped from the
# main dashboard. This tab tracks their usage separately using the same weekly
# heatmap. Driven by the UK_PILOT_SCHOOLS list, not the roster.

def calc_uk_pilots(combined):
    """Weekly login heatmap + summary for the UK pilot schools.
    Not roster-gated — selects rows whose School name is in UK_PILOT_SCHOOLS.
    """
    if combined is None or combined.empty:
        return None

    df = combined.copy()
    df['SchoolNorm'] = df['School'].apply(_norm_name)
    uk_norm = {_norm_name(s) for s in UK_PILOT_SCHOOLS}
    df = df[df['SchoolNorm'].isin(uk_norm)]
    if df.empty:
        return {'weeks': [], 'schools': [], 'totals': {
            'schools_tracked': len(UK_PILOT_SCHOOLS), 'total_logins': 0,
            'weeks_of_data': 0, 'active_recently': 0, 'dormant': 0}}

    df['WeekStart'] = (df['Date'] - pd.to_timedelta(df['Date'].dt.weekday, unit='D')).dt.normalize()

    # Anchor weeks to the FULL dataset span so the UK heatmap columns line up
    # with the rest of the dashboard's timeline.
    all_weeks = (combined['Date'] - pd.to_timedelta(combined['Date'].dt.weekday, unit='D')).dt.normalize()
    weeks_sorted = sorted(all_weeks.unique())
    weeks_iso    = [pd.Timestamp(w).strftime('%Y-%m-%d') for w in weeks_sorted]
    latest_week  = pd.Timestamp(weeks_sorted[-1])

    base = total_logins(df)
    per_week = (base.groupby(['School', 'WeekStart'])['Email'].count()
                    .reset_index().rename(columns={'Email': 'logins'}))

    schools_out = []
    for school, rows in base.groupby('School'):
        wk = per_week[per_week['School'] == school]
        d_map = {pd.Timestamp(w).strftime('%Y-%m-%d'): int(n)
                 for w, n in zip(wk['WeekStart'], wk['logins'])}
        tl = sum(d_map.values())
        uw = len(d_map)
        users = rows.drop_duplicates(subset=['Email'])
        tc = users[users['UserRole'].isin(_TEACHER_ROLES)]['Email'].nunique()
        sc = users[users['UserRole'].isin(_PARTICIPANT_ROLES)]['Email'].nunique()
        last_week = pd.Timestamp(rows['WeekStart'].max())
        weeks_since_last = int((latest_week - last_week).days // 7)
        last_seen = strf(pd.Timestamp(rows['Date'].max()), '%-d %b %Y')
        schools_out.append({
            's': school, 'tl': tl, 'uw': uw, 'tc': int(tc), 'sc': int(sc),
            'last': last_seen, 'dormant': weeks_since_last >= 6,
            'd': d_map,
        })

    schools_out.sort(key=lambda x: (-x['tl'], x['s'].lower()))

    return {
        'weeks': weeks_iso,
        'schools': schools_out,
        'totals': {
            'schools_tracked': len(schools_out),
            'total_logins':    sum(s['tl'] for s in schools_out),
            'weeks_of_data':   len(weeks_iso),
            'active_recently': sum(1 for s in schools_out if not s['dormant']),
            'dormant':         sum(1 for s in schools_out if s['dormant']),
        },
        'date_range_label': f"{strf(pd.Timestamp(combined['Date'].min()), '%-d %b')} – "
                            f"{strf(pd.Timestamp(combined['Date'].max()), '%-d %b %Y')}",
    }


def build_uk_pilots_html(p):
    """Visible HTML block for the UK Pilots tab (between UK_PILOTS_BLOCK markers)."""
    if not p or not p['schools']:
        return ('<h2 class="section-title pacific">🇬🇧 UK Pilots</h2>\n'
                '<p class="section-desc">No UK pilot activity found in the snapshots yet.</p>')
    t = p['totals']
    return f'''<h2 class="section-title pacific">🇬🇧 UK Pilots — {p['date_range_label']}</h2>
        <p class="section-desc">Non-paying UK pilot schools (not in ActiveCampaign Pipeline 6) · tracked separately from paying customers</p>

        <div class="pt-summary-grid">
            <div class="pt-summary-card"><div class="pt-card-top" style="background:var(--lapis)"></div><div class="pt-card-num">{t['schools_tracked']}</div><div class="pt-card-label">UK pilot schools</div></div>
            <div class="pt-summary-card"><div class="pt-card-top" style="background:var(--sky)"></div><div class="pt-card-num">{t['total_logins']}</div><div class="pt-card-label">Total logins</div></div>
            <div class="pt-summary-card"><div class="pt-card-top" style="background:var(--green)"></div><div class="pt-card-num">{t['active_recently']}</div><div class="pt-card-label">Active (last 6 wks)</div></div>
            <div class="pt-summary-card"><div class="pt-card-top" style="background:#b91c1c"></div><div class="pt-card-num" style="color:#b91c1c">{t['dormant']}</div><div class="pt-card-label">Dormant 6+ wks</div></div>
        </div>

        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;flex-wrap:wrap;gap:0.5rem;">
            <div style="font-size:0.8rem;color:var(--gray);">Showing <strong>{len(p['schools'])}</strong> UK pilot schools &nbsp;·&nbsp; Heatmap: {p['date_range_label']}</div>
            <div style="display:flex;gap:12px;font-size:0.75rem;color:var(--gray);align-items:center;">
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#0F6E56;vertical-align:middle;margin-right:3px;"></span>High</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#5DCAA5;vertical-align:middle;margin-right:3px;"></span>Med</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#c8eed9;vertical-align:middle;margin-right:3px;"></span>Low</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#E8E4DF;vertical-align:middle;margin-right:3px;"></span>None</span>
            </div>
        </div>

        <div class="pt-table-header">
            <div>School</div>
            <div id="uk-week-headers" style="display:flex;gap:3px;"></div>
            <div>Last seen</div>
        </div>
        <div class="pt-school-list" id="uk-school-list"></div>'''


def build_uk_pilots_js(p):
    """JS data block for the UK Pilots heatmap (between UK_PILOTS_DATA markers)."""
    if not p:
        return 'var UK_WEEKS=[];var UK_SCHOOLS=[];'
    return (f"var UK_WEEKS={json.dumps(p['weeks'])};\n"
            f"var UK_SCHOOLS={json.dumps(p['schools'], ensure_ascii=False)};")


# ── End-of-run "what happened" report ──────────────────────────────────────────

def build_daily_report(combined):
    """Build the human-readable report of exactly what this run loaded, skipped,
    and dropped — so the operator can tell a good run from a bad one at a glance.
    Returns the report as a string; also identifies dropped snapshot schools.
    """
    lines = []
    add = lines.append

    # ── Roster status — the single most impactful failure mode ──
    # If paying_schools.json failed to load, the whole pipeline silently drops
    # to legacy mode (old EXCLUDED_SCHOOLS list instead of the real AC roster),
    # which changes which schools count as "paying" without any error anywhere
    # else. This banner is deliberately the first thing in the report.
    if _LOAD_REPORT['roster_error']:
        add("=" * 60)
        add("🚨 AC ROSTER FAILED TO LOAD — RUNNING IN DEGRADED LEGACY MODE")
        add("=" * 60)
        add(f"  {_LOAD_REPORT['roster_error']}")
        add("")
        add("  Every number below is computed WITHOUT the ActiveCampaign roster —")
        add("  using the old EXCLUDED_SCHOOLS list instead. School counts, 'paying'")
        add("  status, and totals will NOT match ActiveCampaign until this is fixed.")
        add("")
        add("  FIX: re-run update_sales_dashboard.py, or restore paying_schools.json")
        add("  from the last good git commit, then re-run this script.")
        add("=" * 60)
        add("")

    # ── Files ──
    loaded = _LOAD_REPORT['loaded']
    add("FILES")
    add(f"  Loaded OK ........... {len(loaded)}")

    skipped_total = (len(_LOAD_REPORT['skipped_no_date'])
                     + len(_LOAD_REPORT['skipped_before_wk1'])
                     + len(_LOAD_REPORT['skipped_no_sheet'])
                     + len(_LOAD_REPORT['skipped_missing_cols'])
                     + len(_LOAD_REPORT['errors']))
    add(f"  Skipped / failed .... {skipped_total}")

    def _list(label, items):
        if items:
            add("")
            add(f"  ⚠️  {label} ({len(items)}) — these did NOT reach the dashboard:")
            for it in items:
                add(f"        • {it}")

    _list("No date in filename (rename to 'Daily Usage Snapshot - DD-MM-YYYY.xlsx')",
          _LOAD_REPORT['skipped_no_date'])
    _list("Dated before Week 1 (19 Jan 2026)", _LOAD_REPORT['skipped_before_wk1'])
    _list("No usable data sheet", _LOAD_REPORT['skipped_no_sheet'])
    _list("Missing 'School Name' / 'Email' column — re-export this file",
          _LOAD_REPORT['skipped_missing_cols'])
    if _LOAD_REPORT['errors']:
        add("")
        add(f"  ⚠️  Errors while reading ({len(_LOAD_REPORT['errors'])}):")
        for fn, err in _LOAD_REPORT['errors']:
            add(f"        • {fn}  →  {err}")

    if _LOAD_REPORT['header_typos']:
        add("")
        add(f"  🔤 Header spelling typo caught automatically ({len(_LOAD_REPORT['header_typos'])}) "
            f"— file still loaded fine:")
        for fn, canonical, actual in _LOAD_REPORT['header_typos']:
            add(f"        • {fn}: \"{actual}\" read as {canonical}")
        add("     These are safe — matched with a very high confidence margin. If one")
        add("     looks wrong, re-export that file properly rather than relying on this.")

    # ── Freshness ──
    add("")
    add("FRESHNESS")
    if loaded:
        newest = max(d for _, d, _ in loaded)
        age = (datetime.now().date() - newest.date()).days
        flag = "  ⚠️  STALE — is today's file in the folder?" if age >= 3 else ""
        add(f"  Newest snapshot loaded: {strf(newest, '%A, %-d %B %Y')}  "
            f"({age} day(s) old){flag}")
    else:
        add("  ⚠️  NO FILES LOADED AT ALL — nothing to build from.")

    # ── Schools dropped by the AC-roster gate ──
    dropped = []
    if combined is not None and not combined.empty and _ROSTER_LOOKUP is not None:
        for name in sorted(combined['School'].dropna().unique()):
            if resolve_to_roster(name) is None:
                n_rows = int((combined['School'] == name).sum())
                dropped.append((name, n_rows))
        dropped.sort(key=lambda x: -x[1])

    # ── Spelling drift caught automatically this run ──
    add("")
    add("FUZZY-MATCHED THIS RUN (spelling didn't match AC exactly, matched anyway)")
    if _FUZZY_MATCH_LOG:
        for snap_name, matched_display in sorted(_FUZZY_MATCH_LOG.items()):
            add(f"        • \"{snap_name}\"  →  {matched_display}")
        add("")
        add("     These were counted correctly — no logins were lost. If a mapping")
        add("     above looks wrong, tighten it up by adding an explicit entry to")
        add("     SCHOOL_NAME_ALIASES so it's exact instead of guessed next time.")
    else:
        add("        None this run — every match was exact.")

    investigate = [(n, c) for n, c in dropped if n.strip().lower() not in KNOWN_NON_PAYING]
    expected    = [(n, c) for n, c in dropped if n.strip().lower() in KNOWN_NON_PAYING]

    add("")
    add("SCHOOLS IN SNAPSHOTS BUT NOT ON THE DASHBOARD")
    add("  (name didn't match any AC roster entry)")
    add("")
    add("  ⚠️  INVESTIGATE — not a known exclusion:")
    if investigate:
        for name, n in investigate:
            add(f"        • {name:<40} ({n} login row(s))")
        add("")
        add("     → If any ARE paying schools: move the deal into AC Pipeline 6")
        add("       (Onboarding/Activated, ZAR), or add a SCHOOL_NAME_ALIASES entry")
        add("       for a spelling mismatch. If not paying, add to KNOWN_NON_PAYING.")
    else:
        add("        None — nothing unexpected was dropped. ✅")

    if expected:
        add("")
        add(f"  Known non-paying (UK music services / pilots — correctly excluded), "
            f"{len(expected)}:")
        for name, n in expected:
            add(f"        • {name:<40} ({n} login row(s))")

    return "\n".join(lines), dropped


def print_and_save_report(combined):
    report, dropped = build_daily_report(combined)
    banner = "=" * 60
    print("\n" + banner)
    print("📋 DAILY LOAD & MATCH REPORT")
    print(banner)
    print(report)
    print(banner)
    try:
        stamp = strf(datetime.now(), '%A, %-d %B %Y at %H:%M')
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"Ear Academy — Daily Load & Match Report\nGenerated: {stamp}\n\n")
            f.write(report + "\n")
        print(f"📝 Saved to {REPORT_FILE}")
    except OSError as e:
        print(f"  ⚠️  Could not write {REPORT_FILE}: {e}")
    return dropped


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🎵 Ear Academy Dashboard Updater")
    print("=" * 60)

    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir()
        print(f"Created {DATA_FOLDER}/ — add your Excel files there!")
        return

    # Load the AC paying-schools roster first — this becomes the canonical
    # list of who counts as a "paying school". If missing, we fall back to
    # the legacy snapshot-only behaviour.
    global _ROSTER, _ROSTER_LOOKUP, _DISPLAY_NAME_FOR_DEAL, TOTAL_CUSTOMERS
    _ROSTER, _ROSTER_LOOKUP = load_paying_schools_roster()
    if _ROSTER:
        TOTAL_CUSTOMERS = len(_ROSTER)
        print(f"📋 AC roster loaded: {TOTAL_CUSTOMERS} paying schools "
              f"(from {ROSTER_FILE})")
    else:
        print(f"⚠️  No AC roster at {ROSTER_FILE} — running in legacy mode.")
        print("    Run update_sales_dashboard.py first to generate it.")

    print(f"\n📂 Loading files from {DATA_FOLDER}/...\n")
    combined = load_all_data()

    if combined is None or combined.empty:
        print("❌ No data loaded.")
        return

    # Build the deal-id → canonical-display-name map BEFORE the first
    # paying() call. This is what de-duplicates name variants so a school
    # appearing in snapshots under multiple spellings collapses to one row.
    _DISPLAY_NAME_FOR_DEAL = build_display_name_for_deal(combined)

    pay = paying(combined)
    if _ROSTER:
        active_count = pay['School'].nunique()
        # Resolve to deal_ids to count unique roster schools observed
        active_roster_ids = {(resolve_to_roster(s) or {}).get('deal_id')
                              for s in pay['School'].unique()}
        active_roster_ids.discard(None)
        silent_count = TOTAL_CUSTOMERS - len(active_roster_ids)
        print(f"\n✅ {total_logins(combined)['Email'].count()} unique login events loaded")
        print(f"   Paying schools (AC roster):     {TOTAL_CUSTOMERS}")
        print(f"   ├─ active in snapshots:         {len(active_roster_ids)}")
        print(f"   └─ silent (no snapshot yet):    {silent_count}")
        print(f"   Snapshot rows kept after gate:  {len(pay)}")
        print(f"   Weeks covered:                  {sorted(combined['Week'].unique())}\n")
    else:
        TOTAL_CUSTOMERS = pay['School'].nunique()
        print(f"\n✅ {total_logins(combined)['Email'].count()} unique login events loaded")
        print(f"   Paying schools: {pay['School'].nunique()}")
        print(f"   Weeks covered:  {sorted(combined['Week'].unique())}\n")

    # Compute all metrics
    dp       = calc_daily_pulse(combined)
    snap     = calc_weekly_snapshot(combined)
    patterns = calc_patterns(combined, snap)
    w_stats, last6 = calc_weekly_trends(combined)
    top10         = calc_top10(combined)
    lifetime_data = calc_lifetime_logins(combined)
    usage_patterns = calc_usage_patterns(combined)
    uk_pilots      = calc_uk_pilots(combined)

    # Build HTML sections
    daily_pulse_html  = build_daily_pulse_html(dp)
    weekly_snap_html  = build_weekly_snapshot_html(snap)
    patterns_html     = build_patterns_html(patterns, snap)
    trends_html       = build_trends_html(w_stats, last6)
    top10_html        = build_top10_html(top10)
    lifetime_html     = build_lifetime_logins_html(lifetime_data)
    pt_block_html     = build_usage_patterns_html(usage_patterns)
    pt_data_js        = build_usage_patterns_js(usage_patterns)
    uk_block_html     = build_uk_pilots_html(uk_pilots)
    uk_data_js        = build_uk_pilots_js(uk_pilots)

    updated_date = strf(combined['Date'].max(), '%-d %B %Y')

    if not OUTPUT_FILE.exists():
        print(f"❌ {OUTPUT_FILE} not found.")
        return

    with open(OUTPUT_FILE, 'r') as f:
        html = f.read()

    new_content = (
        daily_pulse_html + '\n'
        + weekly_snap_html + '\n'
        + patterns_html + '\n'
        + trends_html + '\n'
        + top10_html + '\n'
        + lifetime_html
    )

    html = re.sub(
        r'<!-- DASHBOARD_START -->.*?<!-- DASHBOARD_END -->',
        f'<!-- DASHBOARD_START -->{new_content}\n        <!-- DASHBOARD_END -->',
        html, flags=re.DOTALL,
    )
    # Usage Patterns: visible HTML block (summary tiles + filter pills + heatmap shell)
    html = re.sub(
        r'<!-- PATTERNS_BLOCK_START -->.*?<!-- PATTERNS_BLOCK_END -->',
        ('<!-- PATTERNS_BLOCK_START -->\n        '
         '<!-- Auto-generated by update_dashboard.py — do not edit by hand. -->\n'
         f'{pt_block_html}\n'
         '        <!-- PATTERNS_BLOCK_END -->'),
        html, flags=re.DOTALL,
    )
    # Usage Patterns: JS data (WEEKS, SCHOOLS, PC, DESC arrays)
    html = re.sub(
        r'// PATTERNS_DATA_START.*?// PATTERNS_DATA_END',
        f'// PATTERNS_DATA_START\n{pt_data_js}\n// PATTERNS_DATA_END',
        html, flags=re.DOTALL,
    )
    # UK Pilots: visible HTML block
    html = re.sub(
        r'<!-- UK_PILOTS_BLOCK_START -->.*?<!-- UK_PILOTS_BLOCK_END -->',
        ('<!-- UK_PILOTS_BLOCK_START -->\n        '
         '<!-- Auto-generated by update_dashboard.py — do not edit by hand. -->\n'
         f'{uk_block_html}\n'
         '        <!-- UK_PILOTS_BLOCK_END -->'),
        html, flags=re.DOTALL,
    )
    # UK Pilots: JS data (UK_WEEKS, UK_SCHOOLS)
    html = re.sub(
        r'// UK_PILOTS_DATA_START.*?// UK_PILOTS_DATA_END',
        f'// UK_PILOTS_DATA_START\n{uk_data_js}\n// UK_PILOTS_DATA_END',
        html, flags=re.DOTALL,
    )
    html = re.sub(
        r'Updated <span id="lastUpdated">[^<]*</span>',
        f'Updated <span id="lastUpdated">{updated_date}</span>',
        html,
    )

    with open(OUTPUT_FILE, 'w') as f:
        f.write(html)

    print("📊 Summary")
    print(f"   Daily Pulse   : {dp.get('y_logins',0)} logins · {dp.get('y_schools',0)} schools "
          f"({dp.get('y_cls',0)} cls / {dp.get('y_ins',0)} ins)")
    print(f"   Week {snap['max_week']} snapshot: {snap['cw_logins']} logins · {snap['cw_schools']} schools")
    print(f"   Classroom     : {snap['cw_cls_logins']} logins / {snap['cw_cls_schools']} schools")
    print(f"   Instrumental  : {snap['cw_ins_logins']} logins / {snap['cw_ins_schools']} schools")
    print(f"   Consistent    : {snap['consistent_count']} schools (3+ days)")
    print(f"   Quiet 7-13 days: {snap['quiet_7_count']} schools")
    print(f"   Quiet 30+ days : {snap['quiet_30_count']} schools")
    print(f"   New this week : {len(patterns['new_this_week'])}")
    print(f"   Lifetime graph: {len(lifetime_data)} schools ranked")
    if usage_patterns:
        ut = usage_patterns['totals']
        pcnt = usage_patterns['pattern_counts']
        print(f"   Usage Patterns: {ut['schools_tracked']} schools · "
              f"{ut['total_logins']} logins · {ut['weeks_of_data']} weeks")
        for name in PATTERN_ORDER:
            if name in pcnt:
                print(f"      · {name:<24}{pcnt[name]}")
        print(f"      · Gone Quiet (overlay)   {ut['gone_quiet']}")
    print(f"\n✅ Dashboard written → {OUTPUT_FILE}")
    print("=" * 60)

    # Loud, human-readable report of what loaded / skipped / dropped this run.
    print_and_save_report(combined)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
