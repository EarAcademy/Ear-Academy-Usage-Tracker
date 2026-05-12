path = '/Users/russelnerwich/Desktop/ear-academy-analytics/investor.html'
f = open(path); c = f.read(); f.close()
old = '    <div class="header-badge">Updated 30 Mar 2026</div>\n  </div>\n</header>'
new = old.replace('</div>\n</header>', '</div>\n  <div style="max-width:1400px;margin:0 auto;padding:0 2.5rem;display:flex;gap:0.5rem">\n    <a href="index.html" style="padding:0.4rem 1rem;font-family:Lato;font-size:0.78rem;font-weight:700;color:#666;background:#F5F1ED;border-radius:10px 10px 0 0;text-decoration:none;border:1px solid #E8E4DF;border-bottom:none">📊 Usage Analytics</a>\n    <a href="investor.html" style="padding:0.4rem 1rem;font-family:Lato;font-size:0.78rem;font-weight:700;color:#1d70b8;background:#fff;border-radius:10px 10px 0 0;text-decoration:none;border:1px solid #E8E4DF">💼 Sales Activity</a>\n  </div>\n</header>')
open(path,'w').write(new); print('done' if new != c else 'not changed')
