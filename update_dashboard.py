#!/usr/bin/env python3
"""
Ear Academy Usage Analytics - Dashboard Updater
Rebuilds the dashboard with daily-resolution data.
"""

import pandas as pd
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta

# ── Configuration ────────────────────────────────────────────────────────────
DATA_FOLDER     = Path("daily_snapshots")
OUTPUT_FILE     = Path("index.html")
WEEK1_START     = datetime(2026, 1, 19)
TOTAL_CUSTOMERS = 53

# ── Canonical name overrides (exact-match after basic cleaning) ───────────────
# Add entries here only when automated fuzzy matching needs a hard override.
# Format: 'messy raw string' → 'canonical name'
EXACT_OVERRIDES = {
    # HTML-entity variants → clean display name
    'Acudeo Thornview Primary &amp; Secondary School': 'Acudeo Thornview',
    'Acudeo Thornview Primary & Secondary School':     'Acudeo Thornview',
    # Truncated / entity-encoded St Martin variants → single canonical
    "St Martin&#039;s Preparatory Schoo":              'St Martin Preparatory School',
    "St Martin's Preparatory School":                  'St Martin Preparatory School',
    "St Martin&#039;s Preparatory School":             'St Martin Preparatory School',
}

# ── Schools that must NEVER be fuzzy-merged (confirmed distinct schools) ──────
# These are pairs where generic token overlap could create false matches.
# Any name listed here is treated as its own canonical and blocked from merging
# with any other name, regardless of similarity score.
MERGE_BLOCKLIST = {
    'Bay Primary',
    'Plettenberg Bay Christian Primary School',
}

# ── Fuzzy-matching helpers ────────────────────────────────────────────────────

def _clean_for_matching(s):
    """
    Produce a normalised key used purely for fuzzy comparison — NOT for display.
    Steps:
      1. Decode common HTML entities (&amp; &#039; etc.)
      2. Strip accents / unicode noise
      3. Lowercase
      4. Remove punctuation and extra whitespace
      5. Collapse common abbreviations / known typos
    """
    # 1. HTML entities + non-breaking / invisible whitespace
    s = s.replace('\xa0', ' ').replace('\u200b', '').replace('\u2019', "'")
    s = s.replace('&amp;', '&').replace('&#039;', "'").replace('&apos;', "'")
    s = s.replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
    # 2. Unicode normalise → ASCII
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    # 3. Lowercase
    s = s.lower()
    # 4. Strip punctuation / extra spaces
    s = re.sub(r"['\-–—,.]", ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # 5. Spelling normalisations
    s = re.sub(r'\bprepatory\b', 'preparatory', s)   # common typo
    s = re.sub(r'\bst\b',        'saint',        s)   # st → saint
    s = re.sub(r'\bprimary\b',   'primary',      s)
    s = re.sub(r'\bschoo\b',     'school',       s)   # truncation
    s = re.sub(r'\bschool\b',    'school',       s)
    return s


def _token_overlap(a, b):
    """
    Jaccard-style overlap between the two token sets.
    Using union (Jaccard) rather than 'fraction of shorter' prevents
    a 2-word name from falsely absorbing a 6-word name just because
    both happen to share generic tokens like 'primary' or 'school'.
    """
    ta = set(_clean_for_matching(a).split())
    tb = set(_clean_for_matching(b).split())
    if not ta or not tb:
        return 0.0
    # Strip generic stop-words before scoring so "primary school" alone
    # doesn't drive false matches
    stopwords = {'school', 'primary', 'secondary', 'college', 'academy',
                 'the', 'of', 'and', 'saint', 'high', 'preparatory'}
    ta_sig = ta - stopwords or ta   # fall back to full set if all are stopwords
    tb_sig = tb - stopwords or tb
    intersection = len(ta_sig & tb_sig)
    union_       = len(ta_sig | tb_sig)
    return intersection / union_ if union_ else 0.0


def build_canonical_map(raw_names, threshold=0.80):
    """
    Given a list of raw school names (with duplicates reflecting frequency),
    group names that are very likely the same school and return a dict
    {raw_name → canonical_name}.

    Algorithm:
      - Build groups where token-overlap ≥ threshold AND the overlap is
        symmetric (both directions), which prevents short names like "Bay Primary"
        accidentally absorbing unrelated longer names.
      - Canonical = most frequently seen raw name; length breaks ties.
    """
    # Count frequencies
    from collections import Counter
    freq   = Counter(raw_names)
    unique = list(freq.keys())

    # Union-Find
    parent = {n: n for n in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, a in enumerate(unique):
        for b in unique[i+1:]:
            # Never merge names that are on the confirmed-distinct blocklist
            if a in MERGE_BLOCKLIST or b in MERGE_BLOCKLIST:
                continue
            overlap_ab = _token_overlap(a, b)
            if overlap_ab >= threshold:
                union(a, b)

    # Build groups
    groups = {}
    for name in unique:
        root = find(name)
        groups.setdefault(root, []).append(name)

    # Pick canonical = most frequent; length breaks ties
    canonical_map = {}
    for members in groups.values():
        canonical = max(members, key=lambda n: (freq[n], len(n)))
        for m in members:
            canonical_map[m] = canonical

    return canonical_map


# ── Normalise a single raw name ───────────────────────────────────────────────

def normalize_school_name(name, canonical_map=None):
    """
    Two-stage cleaning:
      1. Exact override table (for HTML entities, known abbreviations).
      2. Fuzzy canonical map built dynamically from all names in the dataset.
    """
    if pd.isna(name):
        return ""
    # Strip non-breaking spaces and other invisible characters before anything else
    s = str(name).replace('\xa0', ' ').replace('\u200b', '').replace('\u2019', "'").strip()
    # Stage 1 – exact overrides
    s = EXACT_OVERRIDES.get(s, s)
    # Stage 2 – fuzzy canonical (populated during load_all_data)
    if canonical_map and s in canonical_map:
        s = canonical_map[s]
    return s


def parse_date(filename):
    """Extract a date from messy filenames like 'Daily Usage Snapshot - 20 - 02 - 2026.xlsx'"""
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
    end   = start + timedelta(days=4)          # Mon–Fri
    if start.month == end.month:
        return f"Week {week_num} ({start.strftime('%-d')}–{end.strftime('%-d %b')})"
    return f"Week {week_num} ({start.strftime('%-d %b')}–{end.strftime('%-d %b')})"


def pct_change_html(new_val, old_val, unit=""):
    """Return an HTML badge string for a numeric change."""
    if old_val == 0:
        return '<span class="delta new">new</span>'
    diff = new_val - old_val
    pct  = round((diff / old_val) * 100)
    if diff > 0:
        return f'<span class="delta up">▲ {pct}%</span>'
    if diff < 0:
        return f'<span class="delta down">▼ {abs(pct)}%</span>'
    return '<span class="delta flat">→ same</span>'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_data():
    excel_files = sorted(DATA_FOLDER.glob("*.xlsx"))
    if not excel_files:
        return None

    # ── Pass 1: collect every raw school name seen across all files ───────────
    raw_name_pool = []
    for file_path in excel_files:
        file_date = parse_date(file_path.name)
        if not file_date or assign_week(file_date) is None:
            continue
        try:
            xl    = pd.ExcelFile(file_path)
            sheet = next((s for s in xl.sheet_names if 'Raw Data' in s), None)
            if not sheet:
                continue
            df         = pd.read_excel(file_path, sheet_name=sheet)
            school_col = next((c for c in df.columns
                               if 'school' in str(c).lower() and 'name' in str(c).lower()), None)
            if school_col:
                for v in df[school_col].dropna().unique():
                    cleaned = EXACT_OVERRIDES.get(str(v).strip(), str(v).strip())
                    raw_name_pool.append(cleaned)
        except Exception:
            pass

    # ── Build fuzzy canonical map from the full name pool ─────────────────────
    canonical_map = build_canonical_map(raw_name_pool, threshold=0.80)

    # ── Audit: print any names that were merged ───────────────────────────────
    merges = {}
    for raw, canon in canonical_map.items():
        if raw != canon:
            merges.setdefault(canon, []).append(raw)

    if merges:
        print("\n  🔀 Name merges applied (fuzzy deduplication):")
        for canon, variants in sorted(merges.items()):
            for v in variants:
                print(f"      '{v}'  →  '{canon}'")
        print()
    else:
        print("  ✅ No fuzzy merges needed — all school names are consistent.\n")

    # ── Pass 2: load data, applying canonical map ─────────────────────────────
    rows = []
    for file_path in excel_files:
        file_date = parse_date(file_path.name)
        if not file_date:
            print(f"  ⚠️  Skipped (no date): {file_path.name}")
            continue

        week = assign_week(file_date)
        if week is None:
            print(f"  ⚠️  Skipped (before Week 1): {file_path.name}")
            continue

        try:
            xl    = pd.ExcelFile(file_path)
            sheet = next((s for s in xl.sheet_names if 'Raw Data' in s), None)
            if not sheet:
                print(f"  ⚠️  No 'Raw Data' sheet: {file_path.name}")
                continue

            df = pd.read_excel(file_path, sheet_name=sheet)

            school_col = next((c for c in df.columns
                               if 'school' in str(c).lower() and 'name' in str(c).lower()), None)
            email_col  = next((c for c in df.columns
                               if 'email'  in str(c).lower()), None)
            if not school_col or not email_col:
                print(f"  ⚠️  Missing columns: {file_path.name}")
                continue

            df = df.copy()
            df['School'] = df[school_col].apply(
                lambda n: normalize_school_name(n, canonical_map))
            df['Email']  = df[email_col]
            df['Date']   = file_date
            df['Week']   = week

            # Remove internal / test rows
            mask = ~df['School'].str.contains(
                'Onboarding|Ear Academy|Knowledge Hub', case=False, na=False)
            df = df[mask & (df['School'] != '')]

            rows.append(df[['School', 'Email', 'Date', 'Week']])
            print(f"  ✓ {file_date.strftime('%a %d %b')}  (Week {week})  {len(df):>4} rows  – {file_path.name}")

        except Exception as e:
            print(f"  ⚠️  Error reading {file_path.name}: {e}")

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'])
    return combined


# ── Metric calculations ───────────────────────────────────────────────────────

def calc_daily_pulse(combined):
    """Yesterday vs day-before stats."""
    all_dates = sorted(combined['Date'].dt.date.unique())
    if not all_dates:
        return {}

    yesterday  = all_dates[-1]
    day_before = all_dates[-2] if len(all_dates) >= 2 else None

    y_data  = combined[combined['Date'].dt.date == yesterday]
    y_logins  = len(y_data)
    y_schools = y_data['School'].nunique()

    db_logins  = 0
    db_schools = 0
    if day_before:
        db_data    = combined[combined['Date'].dt.date == day_before]
        db_logins  = len(db_data)
        db_schools = db_data['School'].nunique()

    # Schools that logged in yesterday but NOT on day_before (new that day)
    y_school_set  = set(y_data['School'].unique())
    db_school_set = set()
    if day_before:
        db_school_set = set(
            combined[combined['Date'].dt.date == day_before]['School'].unique())

    new_today = sorted(y_school_set - db_school_set)

    return {
        'yesterday':      yesterday,
        'day_before':     day_before,
        'y_logins':       y_logins,
        'y_schools':      y_schools,
        'db_logins':      db_logins,
        'db_schools':     db_schools,
        'new_schools':    new_today,
    }


def calc_weekly_snapshot(combined):
    max_week  = int(combined['Week'].max())
    prev_week = max_week - 1

    cw = combined[combined['Week'] == max_week]
    pw = combined[combined['Week'] == prev_week] if prev_week >= 1 else combined.iloc[0:0]

    # This week / last week
    cw_logins  = len(cw)
    cw_schools = cw['School'].nunique()
    pw_logins  = len(pw)
    pw_schools = pw['School'].nunique()

    # Schools activated (ever seen, this week change)
    ever_schools     = combined['School'].nunique()
    prev_ever        = combined[combined['Week'] <= prev_week]['School'].nunique() if prev_week >= 1 else 0
    activated_change = ever_schools - prev_ever

    # Consistent users: active on 3+ DAYS this current week
    cw_day_counts = (cw.groupby('School')['Date'].nunique())
    consistent_schools = sorted(cw_day_counts[cw_day_counts >= 3].index.tolist())
    consistent_count   = len(consistent_schools)

    prev_cw_day_counts  = (pw.groupby('School')['Date'].nunique()) if len(pw) else pd.Series(dtype=int)
    prev_consistent     = int((prev_cw_day_counts >= 3).sum())

    # Quiet 7+ days: logged in before but not in last 7 days
    latest_date    = combined['Date'].max()
    cutoff_7       = latest_date - timedelta(days=7)
    active_7       = set(combined[combined['Date'] > cutoff_7]['School'].unique())
    ever           = set(combined['School'].unique())
    quiet_7_schools = sorted(ever - active_7)

    return {
        'max_week':          max_week,
        'prev_week':         prev_week,
        'cw_logins':         cw_logins,
        'cw_schools':        cw_schools,
        'pw_logins':         pw_logins,
        'pw_schools':        pw_schools,
        'ever_schools':      ever_schools,
        'activated_change':  activated_change,
        'consistent_schools': consistent_schools,
        'consistent_count':   consistent_count,
        'prev_consistent':    prev_consistent,
        'quiet_7_schools':   quiet_7_schools,
        'quiet_7_count':     len(quiet_7_schools),
    }


def calc_patterns(combined, snap):
    max_week  = snap['max_week']
    prev_week = snap['prev_week']

    # Drop-offs: active last week, NOT active this week
    cw_schools = set(combined[combined['Week'] == max_week]['School'].unique())
    pw_schools = set(combined[combined['Week'] == prev_week]['School'].unique()) if prev_week >= 1 else set()
    dropoffs   = sorted(pw_schools - cw_schools)

    # New activations this week: never seen before this week
    prior      = set(combined[combined['Week'] < max_week]['School'].unique())
    new_this_week = sorted(cw_schools - prior)

    return {
        'dropoffs':      dropoffs,
        'new_this_week': new_this_week,
    }


def calc_weekly_trends(combined):
    all_weeks = sorted(combined['Week'].unique())
    last6     = all_weeks[-6:]

    stats = {}
    for w in last6:
        wd = combined[combined['Week'] == w]
        stats[w] = {
            'schools': wd['School'].nunique(),
            'logins':  len(wd),
            'label':   week_label(int(w)),
        }
    return stats, last6


def calc_top10(combined):
    """Rank schools by (weeks_active × teachers) + total_logins."""
    grp = combined.groupby('School').agg(
        total_logins =('Email', 'count'),
        teachers     =('Email', 'nunique'),
        weeks_active =('Week',  'nunique'),
    ).reset_index()

    grp['score'] = grp['weeks_active'] * grp['teachers'] * 2 + grp['total_logins']
    grp = grp.sort_values('score', ascending=False).head(10).reset_index(drop=True)

    max_week = int(combined['Week'].max())
    result   = []
    for _, row in grp.iterrows():
        school_weeks = sorted(
            combined[combined['School'] == row['School']]['Week'].unique())
        weeks_str = ', '.join(f"W{int(w)}" for w in school_weeks)
        result.append({
            'name':    row['School'],
            'logins':  int(row['total_logins']),
            'teachers':int(row['teachers']),
            'weeks':   weeks_str,
            'weeks_active': int(row['weeks_active']),
            'in_latest': int(max_week) in [int(w) for w in school_weeks],
        })
    return result


# ── HTML builders ─────────────────────────────────────────────────────────────

def build_daily_pulse_html(dp):
    yesterday_str  = dp['yesterday'].strftime('%A, %-d %B %Y')
    day_before_str = dp['day_before'].strftime('%A, %-d %B') if dp['day_before'] else '–'

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
            <p class="section-desc">Yesterday's activity vs the day before</p>

            <div class="pulse-grid">
                <div class="pulse-card">
                    <div class="pulse-date">{yesterday_str}</div>
                    <div class="pulse-row">
                        <div class="pulse-metric">
                            <div class="pulse-value" id="pulse-logins">{dp['y_logins']}</div>
                            <div class="pulse-label">Logins {login_delta}</div>
                        </div>
                        <div class="pulse-divider"></div>
                        <div class="pulse-metric">
                            <div class="pulse-value" id="pulse-schools">{dp['y_schools']}</div>
                            <div class="pulse-label">Schools {school_delta}</div>
                        </div>
                    </div>
                    <div class="pulse-prev">vs {day_before_str}: {dp['db_logins']} logins · {dp['db_schools']} schools</div>
                    {new_html}
                </div>
            </div>
        </section>'''


def build_weekly_snapshot_html(snap):
    max_week  = snap['max_week']
    prev_week = snap['prev_week']

    logins_delta  = pct_change_html(snap['cw_logins'],  snap['pw_logins'])
    schools_delta = pct_change_html(snap['cw_schools'], snap['pw_schools'])

    act_change = snap['activated_change']
    act_delta  = (f'<span class="delta up">▲ +{act_change} this week</span>' if act_change > 0
                  else f'<span class="delta flat">→ no change</span>')

    cons_delta = pct_change_html(snap['consistent_count'], snap['prev_consistent'])

    # Consistent schools list
    cons_badges = ''.join(f'<span class="school-badge cons-badge">{s}</span>'
                          for s in snap['consistent_schools']) or '<em style="color:var(--gray)">None yet</em>'

    # Quiet schools list
    quiet_badges = ''.join(f'<span class="school-badge quiet-badge">{s}</span>'
                           for s in snap['quiet_7_schools']) or '<em style="color:var(--gray)">All schools active!</em>'

    return f'''
        <!-- ═══════════════════════════════ WEEKLY SNAPSHOT ══════════════════════════════════ -->
        <section class="dashboard-section" id="section-weekly-snapshot">
            <h2 class="section-title pacific">📋 Weekly Snapshot — Week {max_week}</h2>
            <p class="section-desc">This week vs last week · Consistent users · Schools needing attention</p>

            <div class="snapshot-grid">

                <!-- Card 1: This week vs Last week -->
                <div class="snap-card accent-pacific">
                    <div class="snap-label">Week {max_week} vs Week {prev_week}</div>
                    <div class="snap-body">
                        <div class="snap-metric">
                            <span class="snap-value" id="snap-cw-logins">{snap['cw_logins']}</span>
                            <span class="snap-unit">logins {logins_delta}</span>
                        </div>
                        <div class="snap-metric">
                            <span class="snap-value" id="snap-cw-schools">{snap['cw_schools']}</span>
                            <span class="snap-unit">schools {schools_delta}</span>
                        </div>
                    </div>
                    <div class="snap-prev">Last week: {snap['pw_logins']} logins · {snap['pw_schools']} schools</div>
                </div>

                <!-- Card 2: Schools Activated -->
                <div class="snap-card accent-forest">
                    <div class="snap-label">Schools Activated</div>
                    <div class="snap-body">
                        <div class="snap-metric">
                            <span class="snap-value" id="snap-activated">{snap['ever_schools']}</span>
                            <span class="snap-unit">of {TOTAL_CUSTOMERS} {act_delta}</span>
                        </div>
                    </div>
                    <div class="snap-prev">{round(snap['ever_schools']/TOTAL_CUSTOMERS*100)}% of all customers have logged in</div>
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

                <!-- Card 4: Quiet 7+ Days -->
                <div class="snap-card accent-salmon">
                    <div class="snap-label">Quiet 7+ Days</div>
                    <div class="snap-body">
                        <div class="snap-metric">
                            <span class="snap-value" id="snap-quiet">{snap['quiet_7_count']}</span>
                            <span class="snap-unit">schools need attention</span>
                        </div>
                    </div>
                    <div class="snap-badges">{quiet_badges}</div>
                </div>

            </div>
        </section>'''


def build_patterns_html(patterns, snap):
    max_week  = snap['max_week']
    prev_week = snap['prev_week']

    # Drop-offs
    if patterns['dropoffs']:
        drop_badges = ''.join(f'<span class="school-badge quiet-badge">{s}</span>'
                              for s in patterns['dropoffs'])
        drop_html = f'<div class="pattern-count-badge">{len(patterns["dropoffs"])}</div><div class="badge-row">{drop_badges}</div>'
    else:
        drop_html = '<div class="pattern-empty">🎉 No drop-offs — everyone who logged in last week is still active!</div>'

    # New activations
    if patterns['new_this_week']:
        new_badges = ''.join(f'<span class="school-badge new-badge">{s}</span>'
                             for s in patterns['new_this_week'])
        new_html = f'<div class="pattern-count-badge">{len(patterns["new_this_week"])}</div><div class="badge-row">{new_badges}</div>'
    else:
        new_html = '<div class="pattern-empty">No first-time logins this week yet.</div>'

    return f'''
        <!-- ═══════════════════════════════ PATTERNS THIS WEEK ═══════════════════════════════ -->
        <section class="dashboard-section" id="section-patterns">
            <h2 class="section-title forest">🔍 Patterns This Week</h2>
            <p class="section-desc">Movement signals for Week {max_week} vs Week {prev_week}</p>

            <div class="patterns-grid">

                <div class="pattern-block">
                    <div class="pattern-block-title">📉 Drop-offs</div>
                    <div class="pattern-block-desc">Active Week {prev_week}, quiet this week</div>
                    {drop_html}
                </div>

                <div class="pattern-block">
                    <div class="pattern-block-title">🆕 New Activations</div>
                    <div class="pattern-block-desc">First-ever login this week</div>
                    {new_html}
                </div>

                <div class="pattern-block pattern-notes">
                    <div class="pattern-block-title">📝 Notes</div>
                    <div class="pattern-block-desc">Contextual insights for this week</div>
                    <textarea class="notes-field" id="weekly-notes" placeholder="Add your observations here…&#10;e.g. School X mentioned exams, reached out to School Y…"></textarea>
                </div>

            </div>
        </section>'''


def build_trends_html(weekly_stats, last6):
    if not last6:
        return ''

    max_schools = max(weekly_stats[w]['schools'] for w in last6)
    latest_week  = max(last6)
    first_week   = min(last6)
    ls = weekly_stats[latest_week]['schools']
    fs = weekly_stats[first_week]['schools']

    if fs > 0:
        gpct = round(((ls - fs) / fs) * 100)
        if gpct > 0:
            growth_text = f'↗ +{gpct}% vs {weekly_stats[first_week]["label"]}'
        elif gpct < 0:
            growth_text = f'↘ {gpct}% vs {weekly_stats[first_week]["label"]}'
        else:
            growth_text  = f'→ Steady across 6 weeks'
    else:
        growth_text = '📊 First data point'

    bars_html = ''
    for w in reversed(last6):
        schools    = weekly_stats[w]['schools']
        logins     = weekly_stats[w]['logins']
        pct        = int((schools / max_schools) * 100) if max_schools else 0
        is_latest  = ' bar-latest' if w == latest_week else ''
        label      = weekly_stats[w]['label']

        bars_html += f'''
                <div class="week-bar-item">
                    <div class="week-info">
                        <span class="week-label">{label}</span>
                        <span class="week-stats">{schools} schools · {logins} logins</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar bar-schools{is_latest}" style="width:{pct}%" data-value="{schools}"></div>
                    </div>
                </div>'''

    return f'''
        <!-- ════════════════════════════════ WEEKLY TRENDS ═══════════════════════════════════ -->
        <section class="dashboard-section" id="section-trends">
            <h2 class="section-title pacific">📈 Weekly Trends</h2>
            <p class="section-desc">Last {len(last6)} weeks · schools active per week</p>

            <div class="trend-summary">
                <div class="trend-highlight">
                    <div class="trend-label">Latest Week</div>
                    <div class="trend-value" id="thisWeekSchools">{ls} schools</div>
                    <div class="trend-change" id="weekGrowth">{growth_text}</div>
                </div>
            </div>

            <div class="weeks-container" id="weeks-container">
{bars_html}
            </div>
        </section>'''


def build_top10_html(top10, combined):
    max_week = int(combined['Week'].max())
    items_html = ''
    for i, s in enumerate(top10):
        badge_class = 'badge-core' if s['weeks_active'] >= 4 else ('badge-active' if s['in_latest'] else 'badge-quiet')
        badge_text  = ('🏆 Core' if s['weeks_active'] >= 4
                       else ('✅ Active' if s['in_latest'] else '💤 Quiet'))
        items_html += f'''
                <li class="top-school-item">
                    <div class="rank-number">{i+1}</div>
                    <div class="school-info">
                        <div class="school-name">{s['name']}</div>
                        <div class="school-stats">{s['logins']} logins · {s['teachers']} teachers · {s['weeks']}</div>
                    </div>
                    <span class="pattern-badge {badge_class}">{badge_text}</span>
                </li>'''

    return f'''
        <!-- ═══════════════════════════════════ TOP 10 ═══════════════════════════════════════ -->
        <section class="dashboard-section" id="section-top10">
            <h2 class="section-title turkish">🏆 Top 10 Schools</h2>
            <p class="section-desc">Ranked by consistency × teachers × frequency</p>
            <ul class="top-schools-list">{items_html}
            </ul>
        </section>'''


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🎵 Ear Academy Dashboard Updater")
    print("=" * 60)

    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir()
        print(f"Created {DATA_FOLDER}/ — add your Excel files there!")
        return

    print(f"\n📂 Loading files from {DATA_FOLDER}/...\n")
    combined = load_all_data()

    if combined is None or combined.empty:
        print("❌ No data loaded.")
        return

    print(f"\n✅ {len(combined)} rows loaded from "
          f"{combined['School'].nunique()} schools "
          f"across {combined['Week'].nunique()} weeks\n")

    # ── Compute all metrics ───────────────────────────────────────────────────
    dp       = calc_daily_pulse(combined)
    snap     = calc_weekly_snapshot(combined)
    patterns = calc_patterns(combined, snap)
    w_stats, last6 = calc_weekly_trends(combined)
    top10    = calc_top10(combined)

    # ── Build HTML sections ───────────────────────────────────────────────────
    daily_pulse_html    = build_daily_pulse_html(dp)
    weekly_snap_html    = build_weekly_snapshot_html(snap)
    patterns_html       = build_patterns_html(patterns, snap)
    trends_html         = build_trends_html(w_stats, last6)
    top10_html          = build_top10_html(top10, combined)

    updated_date = combined['Date'].max().strftime('%-d %B %Y')

    # ── Read template and inject ──────────────────────────────────────────────
    if not OUTPUT_FILE.exists():
        print(f"❌ {OUTPUT_FILE} not found — cannot inject data.")
        return

    with open(OUTPUT_FILE, 'r') as f:
        html = f.read()

    # Replace the main content block
    new_content = (
        daily_pulse_html
        + '\n'
        + weekly_snap_html
        + '\n'
        + patterns_html
        + '\n'
        + trends_html
        + '\n'
        + top10_html
    )

    html = re.sub(
        r'<!-- DASHBOARD_START -->.*?<!-- DASHBOARD_END -->',
        f'<!-- DASHBOARD_START -->{new_content}\n        <!-- DASHBOARD_END -->',
        html,
        flags=re.DOTALL,
    )

    # Update footer date
    html = re.sub(
        r'Updated <span id="lastUpdated">[^<]*</span>',
        f'Updated <span id="lastUpdated">{updated_date}</span>',
        html,
    )

    with open(OUTPUT_FILE, 'w') as f:
        f.write(html)

    print("📊 Summary")
    print(f"   Daily Pulse  : {dp['y_logins']} logins · {dp['y_schools']} schools on {dp['yesterday']}")
    print(f"   Week {snap['max_week']} snapshot : {snap['cw_logins']} logins · {snap['cw_schools']} schools")
    print(f"   Consistent   : {snap['consistent_count']} schools (3+ days this week)")
    print(f"   Quiet 7+ days: {snap['quiet_7_count']} schools")
    print(f"   Drop-offs    : {len(patterns['dropoffs'])}")
    print(f"   New this week: {len(patterns['new_this_week'])}")
    print(f"\n✅ Dashboard written → {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
