#!/usr/bin/env python3
"""
Ear Academy Usage Analytics - Dashboard Updater
Simple version that updates the dashboard with your daily data.
"""

import pandas as pd
import re
from pathlib import Path
from datetime import datetime

# Configuration
DATA_FOLDER = Path("daily_snapshots")
OUTPUT_FILE = Path("index.html")

def normalize_school_name(name):
    """Clean up school names"""
    if pd.isna(name):
        return ""
    
    mappings = {
        'Acudeo Thornview Primary &amp; Secondary School': 'Acudeo Thornview',
        'Acudeo Thornview Primary & Secondary School': 'Acudeo Thornview',
        'St Martin&#039;s Preparatory School': 'St Martin Preparatory School',
    }
    
    return mappings.get(str(name).strip(), str(name).strip())

def assign_week(filename):
    """Figure out which week a file belongs to"""
    match = re.search(r'(\d{1,2})-(\d{2})-(\d{4})', filename)
    if not match:
        return None

    day, month, year = match.groups()
    date = datetime(int(year), int(month), int(day))

    week1_start = datetime(2026, 1, 19)

    if date < week1_start:
        return None

    return (date - week1_start).days // 7 + 1

def main():
    print("🎵 Ear Academy Dashboard Updater")
    print("=" * 50)
    
    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir()
        print(f"Created {DATA_FOLDER} - add your Excel files there!")
        return
    
    excel_files = sorted(DATA_FOLDER.glob("*.xlsx"))
    if not excel_files:
        print(f"❌ No Excel files in {DATA_FOLDER}")
        return
    
    print(f"\n📂 Processing {len(excel_files)} files...\n")
    
    all_data = []
    for file_path in excel_files:
        week = assign_week(file_path.name)
        if not week:
            continue
        
        try:
            xl = pd.ExcelFile(file_path)
            sheet = None
            for s in xl.sheet_names:
                if 'Raw Data' in s:
                    sheet = s
                    break
            
            if not sheet:
                continue
            
            df = pd.read_excel(file_path, sheet_name=sheet)
            
            # Find columns
            school_col = [c for c in df.columns if 'school' in str(c).lower() and 'name' in str(c).lower()][0]
            email_col = [c for c in df.columns if 'email' in str(c).lower()][0]
            
            clean_df = pd.DataFrame({
                'School': df[school_col].apply(normalize_school_name),
                'Email': df[email_col],
                'Week': week
            })
            
            clean_df = clean_df[~clean_df['School'].str.contains('Onboarding|Ear Academy|Knowledge Hub', case=False, na=False)]
            all_data.append(clean_df)
            
            print(f"✓ Week {week}: {file_path.name}")
            
        except Exception as e:
            print(f"⚠️  Skipped {file_path.name}: {e}")
    
    if not all_data:
        print("❌ No data processed")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Calculate metrics
    total_schools = combined['School'].nunique()
    total_logins = len(combined)
    
    max_week = combined['Week'].max()
    recent_weeks = [max_week, max_week-1, max_week-2, max_week-3]
    recent_weeks = [w for w in recent_weeks if w >= 1]
    
    # School activity by week
    school_weeks = combined.groupby(['School', 'Week']).agg({
        'Email': 'nunique',
        'School': 'count'
    }).rename(columns={'Email': 'Teachers', 'School': 'Logins'}).reset_index()
    
    # Calculate evergreen metrics
    schools_recent = combined[combined['Week'].isin(recent_weeks)]['School'].nunique()
    
    # Regular users: active 3+ of last 4 weeks
    school_week_counts = combined[combined['Week'].isin(recent_weeks)].groupby('School')['Week'].nunique()
    regular_count = (school_week_counts >= 3).sum()
    
    # New this month: only in latest week
    schools_latest = set(combined[combined['Week'] == max_week]['School'].unique())
    schools_earlier = set(combined[combined['Week'] < max_week]['School'].unique())
    new_count = len(schools_latest - schools_earlier)
    
    # Inactive: not in recent weeks
    all_schools = set(combined['School'].unique())
    active_recent_schools = set(combined[combined['Week'].isin(recent_weeks)]['School'].unique())
    inactive_count = len(all_schools - active_recent_schools)
    
    # Multi-teacher: 2+ unique teachers
    teacher_counts = combined.groupby('School')['Email'].nunique()
    multi_teacher_count = (teacher_counts >= 2).sum()
    
    print(f"\n✅ Processed {total_logins} logins from {total_schools} schools")
    print(f"   Active recent: {schools_recent}")
    print(f"   Regular users: {regular_count}")
    print(f"   Multi-teacher: {multi_teacher_count}\n")
    
    # Update HTML
    if not OUTPUT_FILE.exists():
        print(f"❌ {OUTPUT_FILE} not found")
        return
    
    print(f"📝 Updating {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'r') as f:
        html = f.read()
    
    # Update metrics
    html = re.sub(r'<div class="pattern-count" id="pattern-active-recent">\d+</div>',
                  f'<div class="pattern-count" id="pattern-active-recent">{schools_recent}</div>', html)
    html = re.sub(r'<div class="pattern-count" id="pattern-regular">\d+</div>',
                  f'<div class="pattern-count" id="pattern-regular">{regular_count}</div>', html)
    html = re.sub(r'<div class="pattern-count" id="pattern-new-month">\d+</div>',
                  f'<div class="pattern-count" id="pattern-new-month">{new_count}</div>', html)
    html = re.sub(r'<div class="pattern-count" id="pattern-inactive">\d+</div>',
                  f'<div class="pattern-count" id="pattern-inactive">{inactive_count}</div>', html)
    html = re.sub(r'<div class="pattern-count" id="pattern-multi-teacher">\d+</div>',
                  f'<div class="pattern-count" id="pattern-multi-teacher">{multi_teacher_count}</div>', html)
    html = re.sub(r'<div class="pattern-count" id="pattern-total-active">\d+</div>',
                  f'<div class="pattern-count" id="pattern-total-active">{total_schools}</div>', html)
    
    # Update top stats
    html = re.sub(r'<div class="stat-value" id="stat-schools">\d+</div>',
                  f'<div class="stat-value" id="stat-schools">{total_schools}</div>', html)
    html = re.sub(r'<div class="stat-value" id="stat-logins">\d+</div>',
                  f'<div class="stat-value" id="stat-logins">{total_logins}</div>', html)
    
    # Update weekly trends chart data
    print(f"📈 Updating weekly trends bars...")
    
    weekly_stats = {}
    for week in sorted(combined['Week'].unique()):
        week_data = combined[combined['Week'] == week]
        weekly_stats[week] = {
            'schools': week_data['School'].nunique(),
            'logins': len(week_data)
        }
    
    # Get last 6 weeks (or all weeks if less than 6)
    all_weeks = sorted(weekly_stats.keys())
    last_6_weeks = all_weeks[-6:] if len(all_weeks) >= 6 else all_weeks
    
    # Find max schools for percentage calculation
    max_schools = max(weekly_stats[w]['schools'] for w in last_6_weeks)
    
    # Build bars HTML
    bars_html = ""
    for week in reversed(last_6_weeks):  # Most recent first
        schools = weekly_stats[week]['schools']
        logins = weekly_stats[week]['logins']
        percentage = int((schools / max_schools) * 100)
        
        bars_html += f'''
                <div class="week-bar-item">
                    <div class="week-info">
                        <span class="week-label">Week {week}</span>
                        <span class="week-stats">{schools} schools • {logins} logins</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar bar-schools" style="width: {percentage}%" data-value="{schools}"></div>
                    </div>
                </div>'''
    
    # Replace bars in HTML
    html = re.sub(
        r'<div class="weeks-container">.*?</div>\s*</div>\s*\s*<!-- Usage Patterns -->',
        f'<div class="weeks-container">{bars_html}\n            </div>\n        </div>\n        \n        <!-- Usage Patterns -->',
        html,
        flags=re.DOTALL
    )
    
    # Update summary card
    latest_week = max(last_6_weeks)
    first_week = min(last_6_weeks)
    latest_schools = weekly_stats[latest_week]['schools']
    first_schools = weekly_stats[first_week]['schools']
    
    if first_schools > 0:
        growth_pct = int(((latest_schools - first_schools) / first_schools) * 100)
        if growth_pct > 0:
            growth_text = f"↗️ +{growth_pct}% vs Week {first_week}"
        elif growth_pct < 0:
            growth_text = f"↘️ {growth_pct}% vs Week {first_week}"
        else:
            growth_text = f"→ Steady vs Week {first_week}"
    else:
        growth_text = "📊 First data point"
    
    html = re.sub(
        r'<div class="trend-value" id="thisWeekSchools">.*?</div>',
        f'<div class="trend-value" id="thisWeekSchools">{latest_schools} schools</div>',
        html
    )
    
    html = re.sub(
        r'<div class="trend-change" id="weekGrowth">.*?</div>',
        f'<div class="trend-change" id="weekGrowth">{growth_text}</div>',
        html
    )
    
    # Update Week 4 Peak stat
    week4_schools = weekly_stats.get(max_week, {}).get('schools', 0)
    html = re.sub(r'<div class="stat-value" id="stat-week4">\d+</div>',
                  f'<div class="stat-value" id="stat-week4">{week4_schools}</div>', html)
    
    # Calculate Active This Month metric
    print(f"📅 Calculating Active This Month...")
    
    from datetime import datetime
    current_month = datetime.now().strftime('%B')  # e.g., "February"
    month_abbr = current_month[:3]  # "Feb"
    
    # Count unique schools active across all weeks in dataset
    # (All weeks in your data represent the current month)
    active_this_month = total_schools
    
    html = re.sub(r'<div class="stat-value" id="stat-active-month">\d+</div>',
                  f'<div class="stat-value" id="stat-active-month">{active_this_month}</div>', html)
    
    html = re.sub(r'<div class="stat-subtext" id="stat-month-name">schools logged in \w+</div>',
                  f'<div class="stat-subtext" id="stat-month-name">schools logged in {month_abbr}</div>', html)
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(html)
    
    print(f"✅ Dashboard updated!\n")
    print("="*50)
    print("🎉 Done! Open index.html in your browser")
    print("="*50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
