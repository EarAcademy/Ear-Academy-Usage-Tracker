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

import pandas as pd
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta

# ── Configuration ────────────────────────────────────────────────────────────
DATA_FOLDER     = Path("daily_snapshots")
OUTPUT_FILE     = Path("index.html")
WEEK1_START     = datetime(2026, 1, 19)
TOTAL_CUSTOMERS = 53          # total paying customer count

# ── Canonical name overrides ──────────────────────────────────────────────────
EXACT_OVERRIDES = {
    'Acudeo Thornview Primary &amp; Secondary School': 'Acudeo Thornview',
    'Acudeo Thornview Primary & Secondary School':     'Acudeo Thornview',
    "St Martin&#039;s Preparatory Schoo":              'St Martin Preparatory School',
    "St Martin's Preparatory School":                  'St Martin Preparatory School',
    "St Martin&#039;s Preparatory School":             'St Martin Preparatory School',
}

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


def _token_overlap(a, b):
    stopwords = {'school', 'primary', 'secondary', 'college', 'academy',
                 'the', 'of', 'and', 'saint', 'high', 'preparatory'}
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
    """Return 'Paying', 'Demo', or 'Paying' (default for old files)."""
    if pd.isna(bs):
        return 'Paying'
    s = str(bs).strip()
    return s if s in ('Paying', 'Demo') else 'Paying'


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
        return f"Week {week_num} ({start.strftime('%-d')}–{end.strftime('%-d %b')})"
    return f"Week {week_num} ({start.strftime('%-d %b')}–{end.strftime('%-d %b')})"


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
            sheet = next((s for s in xl.sheet_names if 'Raw Data' in s), None)
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

            school_col  = next((c for c in df.columns
                                if 'school' in str(c).lower() and 'name' in str(c).lower()), None)
            email_col   = next((c for c in df.columns
                                if 'email' in str(c).lower()), None)
            product_col = next((c for c in df.columns
                                if 'product' in str(c).lower()), None)
            billing_col = next((c for c in df.columns
                                if 'billing' in str(c).lower()), None)
            role_col    = next((c for c in df.columns
                                if 'role' in str(c).lower()), None)

            if not school_col or not email_col:
                print(f"  ⚠️  Missing core columns: {file_path.name}")
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

        except Exception as e:
            print(f"  ⚠️  Error reading {file_path.name}: {e}")
            import traceback; traceback.print_exc()

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'])
    return combined


# ── Metric calculations ───────────────────────────────────────────────────────

def paying(df):
    return df[df['Billing'] == 'Paying']

def demo(df):
    return df[df['Billing'] == 'Demo']

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

    # Quiet 14+ days (paying)
    latest_date   = pay['Date'].max()
    cutoff_14     = latest_date - timedelta(days=14)
    active_14     = set(pay[pay['Date'] > cutoff_14]['School'].unique())
    ever          = set(pay['School'].unique())
    quiet_14      = sorted(ever - active_14)

    return {
        'max_week': max_week, 'prev_week': prev_week,
        'cw_logins': cw_logins, 'cw_schools': cw_schools,
        'cw_cls_logins': cw_cls_logins, 'cw_ins_logins': cw_ins_logins,
        'cw_cls_schools': cw_cls_schools, 'cw_ins_schools': cw_ins_schools,
        'pw_logins': pw_logins, 'pw_schools': pw_schools,
        'ever_schools': ever_schools, 'activated_change': activated_change,
        'consistent_schools': consistent_schools, 'consistent_count': consistent_count,
        'prev_consistent': prev_consistent,
        'quiet_14_schools': quiet_14, 'quiet_14_count': len(quiet_14),
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

    # Product breakdown per school.
    # Use rows where the file actually had a Product Type column (ProductExplicit=True) to
    # determine each school's real product type.  Then assign cls/ins from total_logins so
    # that W1-W5 logins (files with no Product Type col) are classified correctly instead of
    # being ghost-counted in both classroom AND instrumental via the 'both' fallback expansion.
    explicit_pay = pay[pay['ProductExplicit'] == True]
    school_product_type = {}
    for school, grp_exp in explicit_pay.groupby('School'):
        prods = set(grp_exp['Product'].unique())
        if 'classroom' in prods and 'instrumental' in prods:
            school_product_type[school] = 'both'
        elif 'classroom' in prods:
            school_product_type[school] = 'classroom'
        elif 'instrumental' in prods:
            school_product_type[school] = 'instrumental'

    def _cls_ins(row):
        known = school_product_type.get(row['School'])
        total = row['total_logins']
        if known == 'instrumental':
            return 0, total
        if known == 'classroom':
            return total, 0
        if known == 'both':
            return total, total
        # None → no Product Type data at all; return 0s so values are inert
        return 0, 0

    grp[['cls', 'ins']] = grp.apply(lambda r: pd.Series(_cls_ins(r)), axis=1)
    grp['cls'] = grp['cls'].astype(int)
    grp['ins'] = grp['ins'].astype(int)

    grp['score'] = grp['weeks_active'] * grp['unique_users'] * 2 + grp['total_logins']
    grp = grp.sort_values('score', ascending=False).head(10).reset_index(drop=True)

    result = []
    for _, row in grp.iterrows():
        school_weeks = sorted(pay[pay['School'] == row['School']]['Week'].unique())
        weeks_str    = ', '.join(f"W{int(w)}" for w in school_weeks)
        result.append({
            'name':              row['School'],
            'logins':            int(row['total_logins']),
            'teacher_count':     int(row['teacher_count']),
            'participant_count': int(row['participant_count']),
            'known_product':     school_product_type.get(row['School']),  # None if no data
            'cls':               int(row['cls']),
            'ins':               int(row['ins']),
            'weeks':             weeks_str,
            'weeks_active':      int(row['weeks_active']),
            'in_latest':         max_week in [int(w) for w in school_weeks],
        })
    return result


# ── HTML builders ─────────────────────────────────────────────────────────────

def build_daily_pulse_html(dp):
    if not dp:
        return '<section class="dashboard-section"><p>No data.</p></section>'

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
    act_delta     = (f'<span class="delta up">▲ +{act_change} this week</span>' if act_change > 0
                     else '<span class="delta flat">→ no change</span>')
    cons_delta    = pct_change_html(snap['consistent_count'], snap['prev_consistent'])

    cons_badges = ''.join(f'<span class="school-badge cons-badge">{s}</span>'
                          for s in snap['consistent_schools']) or '<em style="color:var(--gray)">None yet</em>'
    quiet_badges = ''.join(f'<span class="school-badge quiet-badge">{s}</span>'
                           for s in snap['quiet_14_schools']) or '<em style="color:var(--gray)">All schools active!</em>'

    return f'''
        <!-- ═══════════════════════════════ WEEKLY SNAPSHOT ══════════════════════════════════ -->
        <section class="dashboard-section" id="section-weekly-snapshot">
            <h2 class="section-title pacific">📋 Weekly Snapshot — Week {mw}</h2>
            <p class="section-desc">Paying customers only · Week {mw} vs Week {pw}</p>

            <div class="snapshot-grid">

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
                        <div class="snap-metric">
                            <span class="snap-value" id="snap-activated">{snap['ever_schools']}</span>
                            <span class="snap-unit">of {TOTAL_CUSTOMERS} {act_delta}</span>
                        </div>
                    </div>
                    <div class="snap-prev">{round(snap['ever_schools']/TOTAL_CUSTOMERS*100)}% of all paying customers have logged in</div>
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

                <!-- Card 4: Quiet 14+ Days -->
                <div class="snap-card accent-salmon">
                    <div class="snap-label">Quiet 14+ Days</div>
                    <div class="snap-body">
                        <div class="snap-metric">
                            <span class="snap-value" id="snap-quiet">{snap['quiet_14_count']}</span>
                            <span class="snap-unit">schools · no activity in 2+ weeks</span>
                        </div>
                    </div>
                    <div class="snap-badges">{quiet_badges}</div>
                </div>

            </div>
        </section>'''


def build_uk_pilot_html(uk):
    if not uk['has_data']:
        schools_html = '<div class="pilot-empty">No Demo schools have logged in yet.</div>'
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

    pay = paying(combined)
    print(f"\n✅ {total_logins(combined)['Email'].count()} unique login events loaded")
    print(f"   Paying schools: {pay['School'].nunique()}")
    print(f"   Demo schools:   {demo(combined)['School'].nunique()}")
    print(f"   Weeks covered:  {sorted(combined['Week'].unique())}\n")

    # Compute all metrics
    dp       = calc_daily_pulse(combined)
    snap     = calc_weekly_snapshot(combined)
    patterns = calc_patterns(combined, snap)
    w_stats, last6 = calc_weekly_trends(combined)
    uk       = calc_uk_pilot(combined)
    top10    = calc_top10(combined)

    # Build HTML sections
    daily_pulse_html  = build_daily_pulse_html(dp)
    weekly_snap_html  = build_weekly_snapshot_html(snap)
    uk_pilot_html     = build_uk_pilot_html(uk)
    patterns_html     = build_patterns_html(patterns, snap)
    trends_html       = build_trends_html(w_stats, last6)
    top10_html        = build_top10_html(top10)

    updated_date = combined['Date'].max().strftime('%-d %B %Y')

    if not OUTPUT_FILE.exists():
        print(f"❌ {OUTPUT_FILE} not found.")
        return

    with open(OUTPUT_FILE, 'r') as f:
        html = f.read()

    new_content = (
        daily_pulse_html + '\n'
        + weekly_snap_html + '\n'
        + uk_pilot_html + '\n'
        + patterns_html + '\n'
        + trends_html + '\n'
        + top10_html
    )

    html = re.sub(
        r'<!-- DASHBOARD_START -->.*?<!-- DASHBOARD_END -->',
        f'<!-- DASHBOARD_START -->{new_content}\n        <!-- DASHBOARD_END -->',
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
    print(f"   UK Pilot      : {uk['schools']} schools · {uk['logins']} logins")
    print(f"   Consistent    : {snap['consistent_count']} schools (3+ days)")
    print(f"   Quiet 14+ days: {snap['quiet_14_count']} schools")
    print(f"   New this week : {len(patterns['new_this_week'])}")
    print(f"\n✅ Dashboard written → {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
