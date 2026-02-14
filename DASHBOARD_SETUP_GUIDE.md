# Ear Academy Usage Analytics Dashboard
## Setup & Usage Guide

---

## ��� What You're Getting

1. **`ear_academy_dashboard.html`** - Beautiful, interactive dashboard
2. **`update_dashboard.py`** - Script to update the dashboard with new data
3. **This guide** - Step-by-step instructions

---

## 📦 One-Time Setup (5 minutes)

### Step 1: Install Python Requirements

Open your terminal/command prompt and run:

```bash
pip install pandas openpyxl
```

That's it! You only need two packages.

### Step 2: Organize Your Files

Create a folder structure like this:

```
ear-academy-analytics/
├── ear_academy_dashboard.html
├── update_dashboard.py
└── daily_snapshots/
    ├── Daily_Usage_Snapshot_-_19-01-2026.xlsx
    ├── Daily_Usage_Snapshot_-_20-01-2026.xlsx
    ├── Daily_Usage_Snapshot_-_21-01-2026.xlsx
    └── ... (all your daily snapshot files)
```

**Important:** The `daily_snapshots` folder will be created automatically when you first run the script.

---

## 🚀 Daily Workflow (15 seconds!)

### When You Get New Data:

1. **Download** your daily snapshot from your system
2. **Save** it to the `daily_snapshots` folder
3. **Run** the update script:

```bash
python update_dashboard.py
```

4. **Done!** The dashboard is updated.

### Example:

```bash
# Navigate to your folder
cd ~/Documents/ear-academy-analytics

# Run the updater
python update_dashboard.py

# Output:
# 🎵 Ear Academy Dashboard Updater
# ==================================================
# 📂 Found 22 files
# ✓ Week 1: Daily_Usage_Snapshot_-_19-01-2026.xlsx (11 records)
# ✓ Week 2: Daily_Usage_Snapshot_-_26-01-2026.xlsx (15 records)
# ...
# ✅ Dashboard updated: ear_academy_dashboard.html
# 🎉 Done!
```

---

## 🌐 Sharing with Your Team

### Option 1: Google Drive (Recommended)

1. Upload `ear_academy_dashboard.html` to Google Drive
2. Right-click → Get link → Set to "Anyone with the link"
3. Copy the link
4. Share with your team

**When you update:**
- Just replace the file in Google Drive
- The link stays the same
- Everyone sees the latest version automatically!

### Option 2: Dropbox

1. Upload to Dropbox
2. Get shareable link
3. Team uses same link forever

### Option 3: SharePoint / OneDrive

Same concept - upload once, update file, link stays constant.

---

## 📊 Dashboard Features

### What Your Team Sees:

1. **Key Metrics Cards**
   - Schools activated
   - Total logins
   - Core 5 count
   - Week 4 peak

2. **Weekly Trend Chart**
   - Interactive line chart
   - Schools & logins over time
   - Hover for details

3. **Top 10 Schools**
   - Ranked by engagement
   - Pattern badges (Core 5, Bi-weekly, etc.)
   - Login & teacher counts

4. **Usage Patterns Breakdown**
   - Pie chart showing pattern distribution
   - Pattern explanations

5. **Full School Table**
   - Every school, every week
   - Color-coded patterns
   - Sortable by clicking headers

### Mobile Friendly
Works perfectly on phones and tablets!

### Print Friendly
Optimized for printing or saving as PDF for investor decks.

---

## 🎨 Brand Aligned

The dashboard uses:
- ✅ Ear Academy color palette (Pacific, Turkish Delight, etc.)
- ✅ Quicksand & Lato fonts
- ✅ Playful "bouncy circles" design elements
- ✅ Rounded corners throughout
- ✅ Clean, professional aesthetic

Perfect for both **team meetings** and **investor presentations**.

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

**Fix:**
```bash
pip install pandas openpyxl
```

### "No Excel files found"

**Fix:**
Make sure your Excel files are in the `daily_snapshots` folder and have the correct naming format:
- `Daily_Usage_Snapshot_-_DD-MM-YYYY.xlsx`

### "Dashboard not updating"

**Fix:**
1. Make sure `ear_academy_dashboard.html` is in the same folder as `update_dashboard.py`
2. Check that the script ran without errors
3. Refresh your browser (sometimes need to hard refresh: Ctrl+F5 or Cmd+Shift+R)

### Need Help?

Check the script output - it tells you exactly what it's doing and any errors.

---

## 💡 Pro Tips

### Tip 1: Weekly Routine
Set a reminder to run the update script every Friday afternoon after collecting the week's data.

### Tip 2: Backup Your Data
Keep a copy of your `daily_snapshots` folder backed up to cloud storage.

### Tip 3: Version History
If using Google Drive, you can see previous versions of the dashboard (File → Version history).

### Tip 4: Multiple Dashboards
Want separate dashboards for different time periods?
- Rename `ear_academy_dashboard.html` to `month1_dashboard.html`
- Run the script again with different data
- You now have two dashboards!

---

## 📝 File Naming Convention

Your daily snapshots should follow this pattern:
```
Daily_Usage_Snapshot_-_DD-MM-YYYY.xlsx
```

The script automatically determines which week each file belongs to based on the date.

**Week assignments:**
- Week 1: Jan 19-25, 2026
- Week 2: Jan 26-30, 2026  
- Week 3: Feb 2-6, 2026
- Week 4: Feb 9-13, 2026
- Week 5: Feb 16-20, 2026 (auto-updates as you go!)

---

## 🎉 You're All Set!

**Next steps:**
1. Put both files in a folder
2. Create the `daily_snapshots` subfolder
3. Add your Excel files
4. Run `python update_dashboard.py`
5. Open `ear_academy_dashboard.html` in your browser
6. Upload to Google Drive
7. Share link with team

**Your team will love how easy this is!** 🎵

---

Questions? The script provides helpful error messages to guide you.
