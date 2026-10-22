#!/usr/bin/env python3
"""
Ear Academy — Publish the public dashboard site
================================================
Copies ONLY the rendered pages and the JSON they load into the public site
repo (a separate git clone), checks them, then commits and pushes. Raw
exports, trackers, scripts and docs stay in this private pipeline repo.

Usage:
    python publish_site.py            # copy, check, commit, push
    python publish_site.py --no-push  # copy + check + commit locally only

Site clone location: $EAR_SITE_DIR, or ../ear-academy-site next to this repo.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SITE_DIR = Path(os.environ.get('EAR_SITE_DIR', SCRIPT_DIR.parent / 'ear-academy-site'))

PUBLIC_FILES = [
    'index.html',
    'investor.html',
    'pipeline_velocity.html',
    'renewals.html',
    'pipeline_data.json',
    'velocity_data.json',
    'renewals_data.json',
]
SITE_OWNED = {'.git', 'README.md', '.nojekyll'}

EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
RAW_LINK_RE = re.compile(r'\.(xlsx|xls|csv|docx|pdf)\b', re.I)


def check(name, text):
    problems = []
    emails = sorted(set(EMAIL_RE.findall(text)))
    if emails:
        problems.append(f'{len(emails)} email address(es)')
    if RAW_LINK_RE.search(text):
        problems.append('reference to a raw .xlsx/.csv/.docx/.pdf file')
    if name.endswith('.json'):
        try:
            json.loads(text)
        except ValueError as e:
            problems.append(f'invalid JSON ({e})')
    return problems


def git(*args, check_ok=True):
    return subprocess.run(['git', '-C', str(SITE_DIR), *args], check=check_ok,
                          capture_output=True, text=True)


def main(no_push=False):
    if not (SITE_DIR / '.git').exists():
        print(f'❌ Site repo not found at {SITE_DIR}. Clone the public site repo there first.')
        return 1

    failures = {}
    for name in PUBLIC_FILES:
        src = SCRIPT_DIR / name
        if not src.exists():
            failures[name] = ['missing']
            continue
        problems = check(name, src.read_text(encoding='utf-8'))
        if problems:
            failures[name] = problems
    if failures:
        print('🛑 Not published — these files failed the public-data check:')
        for name, problems in failures.items():
            print(f'   • {name}: {", ".join(problems)}')
        return 1

    for item in SITE_DIR.iterdir():
        if item.name in SITE_OWNED or item.name in PUBLIC_FILES:
            continue
        print(f'  🧹 removing non-public file from site: {item.name}')
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    for name in PUBLIC_FILES:
        shutil.copy2(SCRIPT_DIR / name, SITE_DIR / name)
    (SITE_DIR / '.nojekyll').touch()

    git('add', '-A')
    if not git('status', '--porcelain').stdout.strip():
        print('✅ Public site already up to date — nothing to publish.')
        return 0
    git('commit', '-m', 'Update dashboards')
    if no_push:
        print('🛑 --no-push: committed in the site repo, not pushed.')
        return 0
    push = git('push', 'origin', 'main', check_ok=False)
    if push.returncode != 0:
        print(f'❌ Push failed:\n{push.stderr}')
        return 1
    print('✅ Public site published. Live in about a minute.')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-push', action='store_true')
    sys.exit(main(ap.parse_args().no_push))
