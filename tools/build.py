"""Build a portable, searchable static book from local Markdown. No browser CDN."""
from pathlib import Path
import html
import json
import re
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs'
OUT.mkdir(exist_ok=True)
REPO = 'https://github.com/MarinaMackay/kernel-field-notes'
chapters = []
questions = []
for path in sorted((ROOT / 'chapters').glob('*.md')):
    text = path.read_text()
    md = markdown.Markdown(extensions=['extra', 'toc', 'md_in_html', 'sane_lists'])
    body = md.convert(text)
    title = text.splitlines()[0].removeprefix('# ')
    for match in re.finditer(r'<h3 id="([^"]+)">([^<]+)</h3>', body):
        label = html.unescape(match[2])
        if re.match(r'\d+\.\d+ · ', label):
            parts = label.split(' · ', 2)
            questions.append(dict(id=parts[0], difficulty=parts[1], title=parts[2], url=f'{path.stem}.html#{match[1]}', chapter=title))
    chapters.append(dict(slug=path.stem, title=title, body=body, toc=md.toc))

def shell(title, body, active='', toc=''):
    nav = ''.join(f'<a class="{"active" if c["slug"] == active else ""}" href="{c["slug"]}.html"><span>{i:02d}</span>{html.escape(c["title"])}</a>' for i,c in enumerate(chapters,1))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Kernel Field Notes</title><meta name="description" content="A free question-led book on GPU programming, Triton, performance measurement, inference, and kernel learning.">
<link rel="stylesheet" href="assets/book.css"><script src="assets/book.js" defer></script></head>
<body><a class="skip" href="#main">Skip to content</a><header><a class="brand" href="index.html">K<span>/</span>FN <b>Kernel Field Notes</b></a><nav aria-label="Site"><a href="index.html#question-bank">Question bank</a><a href="about.html">About & coverage</a><a href="{REPO}">GitHub ↗</a></nav></header>
<div class="layout"><aside><p class="eyebrow">THE FIELD GUIDE · v0.1</p><a class="start" href="index.html">Start here</a><nav aria-label="Chapters">{nav}</nav><p class="aside-note">Concepts first.<br>Questions next.<br>Understand the trade-offs.</p></aside>
<main id="main">{body}<footer>Free to read · Educational notes, not new benchmark results · <a href="{REPO}/blob/main/CONTRIBUTING.md">Corrections welcome</a></footer></main></div></body></html>'''

for i,c in enumerate(chapters):
    count=sum(q['chapter']==c['title'] for q in questions)
    before=f'<div class="chapter-meta">CHAPTER {i+1:02d} / {len(chapters):02d} <span>{count} answered questions</span></div>'
    prev=f'<a href="{chapters[i-1]["slug"]}.html">← Previous chapter</a>' if i else '<a href="index.html">← Start here</a>'
    nxt=f'<a href="{chapters[i+1]["slug"]}.html">Next chapter →</a>' if i+1<len(chapters) else '<a href="index.html#question-bank">Browse all questions →</a>'
    body=before+c['body']+f'<div class="page-turn">{prev}{nxt}</div>'
    (OUT / (c['slug']+'.html')).write_text(shell(c['title'], body, c['slug']))

cards=''.join(f'<a class="chapter-card" href="{c["slug"]}.html"><span class="number">{i:02d}</span><div><h3>{html.escape(c["title"])}</h3><p>{sum(q["chapter"]==c["title"] for q in questions)} questions · Knowledge, answers & sources</p></div><span aria-hidden="true">↗</span></a>' for i,c in enumerate(chapters,1))
qlinks=''.join(f'<li data-level="{q["difficulty"]}"><a href="{q["url"]}"><span class="qid">{q["id"]}</span><span>{html.escape(q["title"])}</span><span class="badge">{q["difficulty"]}</span></a><small>{html.escape(q["chapter"])}</small></li>' for q in questions)
home=f'''<section class="hero"><p class="eyebrow">A FREE, QUESTION-LED WEB BOOK</p><h1>Learn the concepts.<br><em>Question the details.</em></h1><p class="lead">GPU programming & AI systems — explained through concepts, questions, and worked examples.</p><p class="intro">From threads and memory to Triton, inference, and kernel learning. Read the knowledge first, test your understanding, then open the answer.</p><div class="hero-links"><a class="button" href="01-gpu-foundations.html">Start reading <span>→</span></a><a href="#question-bank">Find a question ↓</a></div><div class="stats"><span><b>{len(chapters)}</b> chapters</span><span><b>{len(questions)}</b> answered questions</span><span><b>3</b> difficulty levels</span></div></section>
<section class="note"><b>A living first edition.</b> These are educational notes with attributed sources and illustrative calculations. No independent GPU experiments are claimed. Some detailed implementations and recent-paper reading records remain under review. <a href="about.html">See coverage and limitations →</a></section>
<section><div class="section-heading"><p class="eyebrow">THE READING PATH</p><h2>Build a mental model.<br>Then put it to work.</h2></div><div class="chapters">{cards}</div></section>
<section id="question-bank"><div class="section-heading"><p class="eyebrow">THE QUESTION BANK</p><h2>What are you trying<br>to understand?</h2></div><div class="search-controls"><label for="search">Search questions and chapter names</label><input id="search" type="search" placeholder="Try occupancy, KV cache, or GRPO…"><label for="level">Difficulty</label><select id="level"><option value="">All levels</option><option>Foundation</option><option>Intermediate</option><option>Advanced</option></select></div><p class="result-count" id="result-count" role="status">{len(questions)} questions</p><ul id="questions" class="question-list">{qlinks}</ul><p id="no-results" hidden>No matching questions. Try a broader term or another difficulty.</p></section>'''
(OUT/'index.html').write_text(shell('Learn GPU programming through questions',home))
about=(ROOT/'ABOUT.md').read_text()
(OUT/'about.html').write_text(shell('About, sources, and coverage', markdown.markdown(about,extensions=['extra'])))
(OUT/'questions.json').write_text(json.dumps(questions, ensure_ascii=False, indent=2))
(OUT/'.nojekyll').touch()
print(f'Built {len(chapters)} chapters, {len(questions)} searchable questions')
