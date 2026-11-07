"""Check question coverage and every local file/fragment in the generated book."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs'
class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.details = 0
        self.main = 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, f'Duplicate ID: {attrs["id"]}'
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == 'details': self.details += 1
        if tag == 'main': self.main += 1

docs = {}
for path in OUT.glob('*.html'):
    doc = Document()
    doc.feed(path.read_text())
    assert doc.main == 1, path
    docs[path.resolve()] = doc
errors=[]
for path, doc in docs.items():
    for link in doc.links:
        u=urlsplit(link)
        if u.scheme or u.netloc: continue
        target=(path.parent/unquote(u.path)).resolve() if u.path else path
        if not target.exists(): errors.append(f'{path.name}: missing {link}')
        elif u.fragment and target in docs and unquote(u.fragment) not in docs[target].ids:
            errors.append(f'{path.name}: missing fragment {link}')
assert not errors, '\n'.join(errors)
index=json.loads((OUT/'questions.json').read_text())
source_ids=[]
for path in (ROOT/'chapters').glob('*.md'):
    source=path.read_text()
    source_ids+=re.findall(r'^### (\d+\.\d+) · ', source, re.M)
    assert '## Knowledge first' in source, path
    assert not re.search(r'notion-file-block:|file://|/Users/|\bddot\b',source,re.I), path
assert len(source_ids)==len(set(source_ids))==102
assert set(source_ids)=={q['id'] for q in index}
assert sum(doc.details for doc in docs.values())==102
assert all(q['difficulty'] in ['Foundation','Intermediate','Advanced'] for q in index)
for q in index:
    u=urlsplit(q['url'])
    assert u.fragment in docs[(OUT/u.path).resolve()].ids, q
print(f'PASS: {len(docs)} pages; 102 unique indexed questions and answers; local links and fragments; no private note references')
