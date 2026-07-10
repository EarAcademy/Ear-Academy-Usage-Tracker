#!/usr/bin/env python3
"""
Newsletter category analysis for Ear Academy.
Reads paying_schools.json + all Daily Usage Snapshot xlsx files.
Outputs newsletter_analysis.json in the same directory.
READ-ONLY analysis — does not modify any existing files.
"""

import json, re, sys, unicodedata
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── Config (mirrors update_dashboard.py) ─────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_FOLDER     = BASE_DIR / "daily_snapshots"
ROSTER_FILE     = BASE_DIR / "paying_schools.json"
OUTPUT_FILE     = BASE_DIR / "newsletter_analysis.json"
WEEK1_START     = datetime(2026, 1, 19)
MAX_SMALL_USERS = 30   # cutoff for "Small But Mighty"

EXCLUDED_SCHOOLS = [
    'Academie Orpheus','Academie Orfeus','Bolton Music Services',
    'Bradford Music and Arts Service','Bury Music','Collingwood College',
    'Salford Community Leisure'
]

EXACT_OVERRIDES = {
    'Acudeo Thornview Primary &amp; Secondary School': 'Acudeo Thornview',
    'Acudeo Thornview Primary & Secondary School':     'Acudeo Thornview',
    "St Martin&#039;s Preparatory Schoo":              'St Martin Preparatory School',
    "St Martin's Preparatory School":                  'St Martin Preparatory School',
    "St Martin&#039;s Preparatory School":             'St Martin Preparatory School',
}

SCHOOL_NAME_ALIASES = {
    "Acudeo Protea Glen":               "Acudeo College Protea Glen",
    "Applewood Preparatory":            "Applewood Preparatory School",
    "CBC Mount Edmund":                 "CBC Mount Edmund (Christian Brothers' College Mount Edmund)",
    "dr.vanderross":                    "Dr. V.D.Ross - C5",
    "Harriston Primary School":         "Harriston School (Primary)",
    "Hermannsburg School":              "Hermannsburg School (Primary)",
    "Herzlia High School":              "Herzlia Renewal 2025",
    "Herzlia Highlands":                "Herzlia Primary",
    "Herzlia Weitzman Primary School":  "Herzlia Weitzman",
    "Holy Cross RC Primary":            "Holy Cross R C Primary",
    "Lebone II College":                "Lebone II College (Primary)",
    "Princess Park College":            "Royal Schools Princess Park",
    "Sky City Primary School":          "Royal Schools Sky City",
    "St Catherines":                    "ST CATHERINE'S DOMINICAN CONVENT- SA",
    "St Martin Preparatory School":     "St Martin's Preparatory School",
    "St Martins Preparatory School":    "St Martin's Preparatory School",
    "Sunvalley Primary School":         "Sun Valley",
    "Trinity House":                    "TrinityHouse",
}

MILESTONES = [100, 250, 500, 1000, 2500, 5000, 10000]

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_date(filename):
    m = re.search(r'(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{4})', filename)
    if not m: return None
    day, month, year = m.groups()
    try: return datetime(int(year), int(month), int(day))
    except ValueError: return None

def assign_week(date):
    if date < WEEK1_START: return None
    return (date - WEEK1_START).days // 7 + 1

def normalize(name):
    if pd.isna(name): return ""
    s = str(name).replace('\xa0',' ').replace('​','').replace('’',"'").strip()
    return EXACT_OVERRIDES.get(s, s)

def classify_product(pt):
    if pd.isna(pt): return 'both'
    pt = str(pt).strip().lower()
    if 'classroom' in pt and 'instrumental' in pt: return 'both'
    if 'classroom' in pt: return 'classroom'
    if 'instrumental' in pt: return 'instrumental'
    return 'both'

def classify_billing(bs):
    if pd.isna(bs): return 'Paying'
    bs = str(bs).strip().lower()
    if 'demo' in bs or 'pilot' in bs: return 'Demo'
    return 'Paying'

def should_exclude(name):
    nl = name.lower()
    for ex in EXCLUDED_SCHOOLS:
        if ex.lower() in nl: return True
    return False

# ── Load roster ───────────────────────────────────────────────────────────────
with open(ROSTER_FILE) as f:
    roster_data = json.load(f)

roster_lookup = {}
for entry in roster_data["schools"]:
    for field in ("title", "account_name"):
        key = (entry.get(field) or "").strip().lower()
        if key and key not in roster_lookup:
            roster_lookup[key] = entry

# Apply explicit aliases into lookup
for snap_name, target in SCHOOL_NAME_ALIASES.items():
    target_entry = roster_lookup.get(target.strip().lower())
    if target_entry:
        roster_lookup[snap_name.strip().lower()] = target_entry

paying_deal_ids = {e["deal_id"] for e in roster_data["schools"]}

def resolve(name):
    return roster_lookup.get(str(name).strip().lower())

def display_name(name):
    """Return a clean display name for a school, stripping internal suffixes."""
    entry = resolve(name)
    if not entry:
        return name
    title = (entry.get("title") or name).strip()
    for suffix in (" - Core Education Group", "-Core Education Group",
                   " - Core Education", "-Core Education", " (Primary)",
                   " Primary School", " School"):
        if title.endswith(suffix):
            title = title[:-len(suffix)].strip()
            break
    if title.isupper():
        title = title.title()
    return title

# ── Load all snapshots ────────────────────────────────────────────────────────
print("Loading snapshots...", flush=True)
all_rows = []
skipped = 0

snap_files = sorted(DATA_FOLDER.glob("*.xlsx"))
for fp in snap_files:
    name = fp.name
    # Skip B2C and UK files
    if name.lower().startswith("b2c") or name.lower().startswith("uk"):
        continue
    file_date = parse_date(name)
    if not file_date:
        skipped += 1
        continue
    week = assign_week(file_date)
    if week is None:
        continue

    try:
        xl = pd.ExcelFile(fp, engine="openpyxl")
        # Find the right sheet
        sheet = next((s for s in xl.sheet_names
                      if any(k in s.lower() for k in
                             ('usage','data','logins','snapshot','report','all','main','schools'))),
                     xl.sheet_names[0] if xl.sheet_names else None)
        if sheet is None: continue
        df = xl.parse(sheet)

        school_col  = next((c for c in df.columns if 'school' in str(c).lower() and 'name' in str(c).lower()), None)
        email_col   = next((c for c in df.columns if 'email'   in str(c).lower()), None)
        product_col = next((c for c in df.columns if 'product' in str(c).lower()), None)
        billing_col = next((c for c in df.columns if 'billing' in str(c).lower()), None)
        role_col    = next((c for c in df.columns if 'role'    in str(c).lower()), None)

        if not school_col or not email_col:
            skipped += 1
            continue

        df = df[[c for c in [school_col, email_col, product_col, billing_col, role_col] if c is not None]].copy()
        df.rename(columns={school_col: 'RawSchool', email_col: 'Email'}, inplace=True)
        df['School']   = df['RawSchool'].apply(normalize)
        df['Date']     = file_date
        df['Week']     = week
        df['Product']  = df[product_col].apply(classify_product) if product_col else 'both'
        df['Billing']  = df[billing_col].apply(classify_billing) if billing_col else 'Paying'
        df['UserRole'] = df[role_col].astype(str).str.strip() if role_col else ''

        # Remove internal/excluded
        mask = (
            ~df['School'].str.contains('Onboarding|Ear Academy|Knowledge Hub', case=False, na=False) &
            (df['School'] != '') & (df['School'] != 'nan') &
            (df['Billing'] == 'Paying') &
            ~df['School'].apply(should_exclude)
        )
        df = df[mask]

        # Only keep paying schools (resolve to roster)
        df = df[df['School'].apply(lambda n: resolve(n) is not None)]

        # Deduplicate (School, Email, Date)
        df = df.drop_duplicates(subset=['School', 'Email', 'Date'])

        # Expand 'both' product rows
        expanded = []
        for _, row in df.iterrows():
            if row['Product'] == 'both':
                expanded.append({**dict(row), 'Product': 'classroom'})
                expanded.append({**dict(row), 'Product': 'instrumental'})
            else:
                expanded.append(dict(row))
        if expanded:
            all_rows.append(pd.DataFrame(expanded))

    except Exception as e:
        print(f"  ⚠ {name}: {e}", flush=True)
        skipped += 1

if not all_rows:
    print("ERROR: No data loaded!", flush=True)
    sys.exit(1)

df_all = pd.concat(all_rows, ignore_index=True)
# Master dedup after expansion
df_all = df_all.drop_duplicates(subset=['School','Email','Date','Product'])

max_week = int(df_all['Week'].max())
max_date = df_all['Date'].max()
print(f"Loaded {len(df_all)} rows across {df_all['School'].nunique()} schools, up to week {max_week} ({max_date.strftime('%d %b %Y')})", flush=True)

# Map school names to display names
name_map = {}
for raw in df_all['School'].unique():
    name_map[raw] = display_name(raw)

df_all['DisplayName'] = df_all['School'].map(name_map)

# ── Per-school lifetime stats ─────────────────────────────────────────────────
# Treat each (School, Email, Date) as ONE login event regardless of product expansion
df_logins = df_all.drop_duplicates(subset=['School','Email','Date'])

school_stats = {}
for school, grp in df_logins.groupby('School'):
    dn = name_map[school]
    lifetime = len(grp)
    unique_users = grp['Email'].nunique()
    first_week = int(grp['Week'].min())
    weeks_active = grp['Week'].nunique()
    total_possible_weeks = max_week - first_week + 1
    pct_weeks_active = weeks_active / total_possible_weeks if total_possible_weeks > 0 else 0

    school_stats[school] = {
        'display': dn,
        'lifetime': lifetime,
        'unique_users': unique_users,
        'lpuu': lifetime / unique_users if unique_users > 0 else 0,
        'first_week': first_week,
        'weeks_active': weeks_active,
        'total_possible': total_possible_weeks,
        'pct_active': pct_weeks_active,
        'avg_per_active_week': lifetime / weeks_active if weeks_active > 0 else 0,
    }

# Per-week logins per school
weekly = df_logins.groupby(['School','Week']).size().reset_index(name='logins')

# ── Category analyses ─────────────────────────────────────────────────────────
results = {}

# ── 1. MILESTONE MOMENT ───────────────────────────────────────────────────────
# Find which milestone each school is nearest to (just below or just above)
# and in which week they crossed it (sort data by week, compute running total)

print("Computing milestone moment...", flush=True)
milestone_candidates = []

for school, grp in df_logins.groupby('School'):
    weekly_counts = grp.groupby('Week').size().sort_index()
    cumulative = weekly_counts.cumsum()

    for milestone in MILESTONES:
        # Find first week cumulative >= milestone
        crossed_weeks = cumulative[cumulative >= milestone]
        if crossed_weeks.empty:
            # Below milestone — how close?
            current = int(cumulative.iloc[-1])
            gap = milestone - current
            pct_away = gap / milestone
            if pct_away <= 0.15:  # within 15% below
                milestone_candidates.append({
                    'school': school, 'display': name_map[school],
                    'milestone': milestone, 'current': current,
                    'status': 'approaching',
                    'gap': gap, 'crossed_week': None,
                    'crossed_recent': False,
                    'sort_key': pct_away
                })
        else:
            # Already crossed
            crossed_week = int(crossed_weeks.index[0])
            current = int(cumulative.iloc[-1])
            milestone_candidates.append({
                'school': school, 'display': name_map[school],
                'milestone': milestone, 'current': current,
                'status': 'crossed',
                'gap': 0, 'crossed_week': crossed_week,
                'crossed_recent': crossed_week >= max_week - 1,
                'sort_key': -1 if crossed_week >= max_week - 1 else current
            })
        break  # one milestone per school (the relevant one)

# Actually, let's find the "relevant" milestone for each school:
# — If they just crossed one in the last 2 weeks, that milestone
# — Otherwise, the milestone they're closest to (highest they haven't hit, or lowest above)
milestone_final = []
for school, grp in df_logins.groupby('School'):
    weekly_counts = grp.groupby('Week').size().sort_index()
    cumulative = weekly_counts.cumsum()
    current = int(cumulative.iloc[-1])

    # Check all milestones they crossed in the last 2 weeks
    recent_crossed = []
    for ms in MILESTONES:
        crossed_weeks = cumulative[cumulative >= ms]
        if not crossed_weeks.empty:
            cw = int(crossed_weeks.index[0])
            if cw >= max_week - 1:
                recent_crossed.append((ms, cw, current))

    if recent_crossed:
        ms, cw, cur = max(recent_crossed)
        milestone_final.append({
            'school': school, 'display': name_map[school],
            'milestone': ms, 'current': cur,
            'status': 'just_crossed', 'crossed_week': cw,
            'crossed_recent': True, 'gap': 0,
            'sort_key': 0
        })
        continue

    # Otherwise: the milestone they're nearest to
    best = None
    best_pct = 999
    for ms in MILESTONES:
        crossed_weeks = cumulative[cumulative >= ms]
        if crossed_weeks.empty:
            pct = (ms - current) / ms
            if pct < best_pct:
                best_pct = pct
                best = {'school': school, 'display': name_map[school],
                        'milestone': ms, 'current': current,
                        'status': 'approaching', 'crossed_week': None,
                        'crossed_recent': False, 'gap': ms - current,
                        'sort_key': pct}
            break  # only look at the next milestone above
        else:
            # They've crossed this one — check if they're between this and next
            pass

    if best:
        if best_pct <= 0.20:  # within 20% of next milestone
            milestone_final.append(best)

# Sort: recently-crossed first, then closest (smallest gap pct)
milestone_final.sort(key=lambda x: (not x['crossed_recent'], x['sort_key']))

top3_milestone = milestone_final[:3]
results['milestone'] = top3_milestone

# ── 2. MOST IMPROVED ──────────────────────────────────────────────────────────
print("Computing most improved...", flush=True)

# Most recent full week = max_week (if at least 3 days of data), else max_week - 1
days_in_max_week = df_logins[df_logins['Week'] == max_week]['Date'].nunique()
recent_week = max_week if days_in_max_week >= 3 else max_week - 1
prior_week  = recent_week - 4

improved = []
for school, stats in school_stats.items():
    if stats['lifetime'] < 10:
        continue
    school_weekly = weekly[weekly['School'] == school].set_index('Week')['logins']
    recent_logins = int(school_weekly.get(recent_week, 0))
    prior_logins  = int(school_weekly.get(prior_week, 0))
    if prior_logins == 0:
        continue
    pct = (recent_logins - prior_logins) / prior_logins * 100
    if pct <= 0:
        continue
    improved.append({
        'school': school, 'display': stats['display'],
        'recent_week': recent_week, 'recent_logins': recent_logins,
        'prior_week': prior_week,   'prior_logins': prior_logins,
        'pct_change': round(pct, 1)
    })

improved.sort(key=lambda x: -x['pct_change'])
results['most_improved'] = improved[:5]

# ── 3. SMALL BUT MIGHTY ───────────────────────────────────────────────────────
print("Computing small but mighty...", flush=True)

small_mighty = []
for school, stats in school_stats.items():
    if stats['unique_users'] >= MAX_SMALL_USERS:
        continue
    if stats['lifetime'] < 5:  # need some activity
        continue
    small_mighty.append({
        'school': school, 'display': stats['display'],
        'unique_users': stats['unique_users'],
        'lifetime': stats['lifetime'],
        'lpuu': round(stats['lpuu'], 1)
    })

small_mighty.sort(key=lambda x: -x['lpuu'])
results['small_mighty'] = small_mighty[:5]

# ── 4. CONSISTENCY AWARD ─────────────────────────────────────────────────────
print("Computing consistency award...", flush=True)

# Find the school with most lifetime logins (to exclude from consistency)
top_volume_school = max(school_stats, key=lambda s: school_stats[s]['lifetime'])
print(f"  Excluding top-volume school: {name_map[top_volume_school]} ({school_stats[top_volume_school]['lifetime']} logins)", flush=True)

consistency = []
for school, stats in school_stats.items():
    if school == top_volume_school:
        continue
    if stats['total_possible'] < 4:  # too new
        continue
    if stats['avg_per_active_week'] < 2:  # filter for genuine engagement
        continue
    consistency.append({
        'school': school, 'display': stats['display'],
        'weeks_active': stats['weeks_active'],
        'total_possible': stats['total_possible'],
        'pct_active': round(stats['pct_active'] * 100, 1),
        'avg_per_active_week': round(stats['avg_per_active_week'], 1),
        'lifetime': stats['lifetime']
    })

consistency.sort(key=lambda x: (-x['pct_active'], -x['avg_per_active_week']))
results['consistency'] = consistency[:5]
results['excluded_from_consistency'] = {
    'school': top_volume_school,
    'display': name_map[top_volume_school],
    'lifetime': school_stats[top_volume_school]['lifetime']
}

# ── 5. WELCOME BACK ───────────────────────────────────────────────────────────
print("Computing welcome back...", flush=True)

welcome_back = []
for school, grp in df_logins.groupby('School'):
    active_weeks = sorted(grp['Week'].unique())
    if not active_weeks: continue

    # Check if active in last 2 weeks
    recent_active = any(w >= max_week - 1 for w in active_weeks)
    if not recent_active: continue

    # Find longest gap before the most recent activity burst
    # Walk through weeks and find a gap of 6+ weeks
    found_gap = False
    for i in range(1, len(active_weeks)):
        gap = active_weeks[i] - active_weeks[i-1] - 1
        if gap >= 6:
            # Check that after this gap there is activity in the last 2 weeks
            post_gap_weeks = [w for w in active_weeks[i:]]
            if any(w >= max_week - 1 for w in post_gap_weeks):
                recent_logins = int(df_logins[(df_logins['School']==school) & (df_logins['Week']>=max_week-1)]['Email'].nunique())
                welcome_back.append({
                    'school': school, 'display': name_map[school],
                    'gap_weeks': gap,
                    'last_active_before_gap': int(active_weeks[i-1]),
                    'returned_week': int(active_weeks[i]),
                    'recent_logins': recent_logins
                })
                found_gap = True
                break

results['welcome_back'] = welcome_back

# ── 6. CLASSROOM SPOTLIGHT ────────────────────────────────────────────────────
print("Computing classroom spotlight...", flush=True)

df_cls = df_all[df_all['Product'] == 'classroom'].drop_duplicates(subset=['School','Email','Date'])
cls_counts = df_cls.groupby('School').size().reset_index(name='logins')
cls_counts['display'] = cls_counts['School'].map(name_map)
cls_counts = cls_counts.sort_values('logins', ascending=False)
results['classroom'] = cls_counts.head(5)[['school','display','logins']].rename(columns={'school':'School'}).to_dict('records')
# Fix: School col was named wrong above
results['classroom'] = cls_counts.head(5).rename(columns={'School':'school'}).to_dict('records')

# ── 7. INSTRUMENTAL SPOTLIGHT ─────────────────────────────────────────────────
print("Computing instrumental spotlight...", flush=True)

df_ins = df_all[df_all['Product'] == 'instrumental'].drop_duplicates(subset=['School','Email','Date'])
ins_counts = df_ins.groupby('School').size().reset_index(name='logins')
ins_counts['display'] = ins_counts['School'].map(name_map)
ins_counts = ins_counts.sort_values('logins', ascending=False)
results['instrumental'] = ins_counts.head(5).rename(columns={'School':'school'}).to_dict('records')

# ── Save results ──────────────────────────────────────────────────────────────
results['_meta'] = {
    'generated_at': datetime.now().isoformat(),
    'max_week': max_week,
    'max_date': max_date.strftime('%Y-%m-%d'),
    'total_schools': len(school_stats),
    'total_rows': len(df_all),
    'top_volume_school': name_map[top_volume_school],
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✅ Results saved to {OUTPUT_FILE}", flush=True)
print(f"   Max week: {max_week} (data through {max_date.strftime('%d %b %Y')})", flush=True)
print(f"   Schools analysed: {len(school_stats)}", flush=True)
