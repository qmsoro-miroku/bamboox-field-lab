"""Build the static browser-AI deployment from the original Jinja templates.
Run from the repository: python scripts/build_public.py
Existing public dataset assets and exported ONNX models are preserved.
"""
from pathlib import Path
import sys, re, json, shutil, argparse
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

REPO=Path(__file__).resolve().parent.parent
DOCS=REPO/'docs'
BASE='/bamboox-field-lab'
PAGES={'intro.html':('','home'),'hub.html':('app/hub','hub'),'index.html':('app','dashboard'),
       'inspection.html':('app/inspection','inspection'),'camera.html':('app/camera','camera'),
       'records.html':('app/records','records'),'labeling.html':('app/labeling','labeling'),'settings.html':('app/settings','settings')}
environment=Environment(loader=FileSystemLoader(str(REPO/'frontend')),undefined=StrictUndefined,autoescape=select_autoescape(['html']))

def public_words(html):
    changes={
        'SPECIES + GROWTH · LOCAL':'SPECIES + GROWTH · BROWSER AI',
        'FIELD LAB / LOCAL WORKSPACE':'FIELD LAB / PUBLIC WEB DEMO',
        'BambooX Field Lab · Local workspace':'BambooX Field Lab · Public web demo',
        'Your local bamboo field workspace.':'Your bamboo field workspace, available online.',
        'Local AI / Local records':'Browser AI / Your private journal',
        'SPECIMEN RECONNAISSANCE / LOCAL AI':'SPECIMEN RECONNAISSANCE / BROWSER AI',
        'locally trained YOLOv8n model':'trained YOLOv8n model running in this browser',
        'local YOLO detections':'on-device YOLO detections',
        'the local journal':'your browser journal',
        'local SQLite':'browser storage',
        'Local SQLite':'Browser journal',
        'SQLite database':'Browser journal',
        'Local database':'Browser journal',
        'SQLite':'browser journal',
        'Local-first:':'Private by default:',
        'local-first workspace':'private browser workspace',
        'Local-first workspace':'Private browser workspace',
        'Local AI · Two scan modes':'Browser AI · Two scan modes',
        'MODEL PREDICTION · SPECIES ONLY':'MODEL PREDICTION · SPECIES',
        'LOCAL CAMERA · TWO AI MODES ACTIVE':'BROWSER CAMERA · TWO AI MODES',
        'run local MobileNetV3-Small':'run MobileNetV3-Small in this browser',
        'local MobileNetV3-Small':'browser MobileNetV3-Small',
        'local records':'your browser records',
        'Local records':'Your browser records',
        'Local image storage':'Browser image storage',
        'LOCAL INSPECTION RECORD':'PRIVATE BROWSER INSPECTION RECORD',
        'Locally generated inspection report':'Browser-generated inspection report',
        'Local SQLite workspace':'Private browser journal',
        'No internet connection or CDN is required.':'The website and models load over HTTPS. AI inference runs on your device.',
        'Your installation runs against files on this computer.':'The public demo runs AI on your device and keeps your journal in this browser.',
        'Inspection images and growth-stage dataset annotations are kept in local project folders.':'Your saved inspection photos stay in this browser. The public dataset is provided separately.',
        'Read-only diagnostics from your running FastAPI server.':'Runtime and storage information from this browser.',
        'FastAPI backend':'Browser AI runtime',
        'Database file path':'Journal location',
        'Local database file path':'Browser journal location',
        'Copy database file path':'Copy journal location',
        'Inspect records and locate the browser journal file.':'Inspect the records saved in this browser.',
        'Inspection photos and the growth-stage YOLO dataset.':'Your browser journal photos and public growth-stage annotations.',
        'Available on disk':'Available for browser download',
        'Not found':'Unavailable',
        'Camera is unavailable. Use localhost in a supported browser, or upload an image.':'Camera access requires HTTPS and a supported browser. You can also upload an image.',
        'This prototype predicts bamboo':'This research demo predicts bamboo',
        'Your local BambooX workspace':'Your private browser journal',
        'your local BambooX workspace':'your private browser journal',
        'local project folders':'this browser',
        'Reading local YOLO annotations':'Reading public YOLO annotations',
        'local system information':'browser system information',
        'local components':'browser components',
        'Local dataset':'Public dataset',
    }
    for before,after in changes.items():html=html.replace(before,after)
    html=re.sub(r'<div class="settings-callout"><strong>Backup reminder:</strong>.*?</div>', '<div class="settings-callout"><strong>Your browser journal:</strong> Export CSV files or printable reports before clearing site data. Records are not synchronized across devices.</div>',html,flags=re.S)
    html=html.replace('image.src = `/api/dashboard/image/${encodeURIComponent(record.id)}`;', 'image.src = record.image_url;')
    html=html.replace('`/api/growth-dataset/image/${item.split}/${encodeURIComponent(item.filename)}`', '`'+BASE+'/data/images/${item.split}/${encodeURIComponent(item.filename)}`')
    html=html.replace('window.location.href =\n      "/api/inspections/export.csv?" + params(false);','window.bambooxDownload("/api/inspections/export.csv?" + params(false), "bamboox_inspections.csv");')
    return html

def public_link(match):
    attr,url=match.groups()
    if url.startswith('//'):return match.group()
    if url=='/':target=BASE+'/'
    elif url.startswith('/app'):
        path,sep,hashpart=url.partition('#');path=path.rstrip('/')+'/'
        target=BASE+path+(sep+hashpart if sep else '')
    else:target=BASE+url
    return f'{attr}="{target}"'

def build():
    (DOCS/'assets').mkdir(parents=True,exist_ok=True)
    for source in (REPO/'frontend/assets').iterdir():
        if source.is_file() and 'backup' not in source.name.lower():shutil.copy2(source,DOCS/'assets'/source.name)
    shutil.copy2(REPO/'scripts/web-api.js',DOCS/'assets/web-api.js')
    for source,(route,active) in PAGES.items():
        html=environment.get_template(source).render(active_page=active)
        html=public_words(html)
        html=re.sub(r'(href|src)="(/[^"\s]*)"',public_link,html)
        runtime=f'<script>window.BAMBOOX_BASE={json.dumps(BASE)};</script>\n<script src="{BASE}/assets/ort/ort.wasm.min.js"></script>\n<script src="{BASE}/assets/web-api.js"></script>\n'
        html=html.replace('<script defer src=',runtime+'<script defer src=',1)
        html=html.replace('</head>',f'<link rel="icon" type="image/svg+xml" href="{BASE}/assets/favicon.svg">\n</head>')
        if active in ('inspection','camera'):
            marker='        <header class="page-header">'
            # Reuse the existing layout; the notice belongs to the scanner page.
            html=html.replace('<div class="fl-scanner-tabs"','<p class="safe-note" id="browserAIStatus">Public demo: AI runs on your device. The first scan downloads the model. Saved species inspections stay in this browser.</p>\n<div class="fl-scanner-tabs"',1) if active=='inspection' else html
        destination=DOCS/route/'index.html';destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(html,encoding='utf-8',newline='\n')
    (DOCS/'.nojekyll').touch()
    (DOCS/'assets/favicon.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#2a5c41"/><g transform="translate(4 4)" fill="none" stroke="#c9ed76" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4C11 3 5 8 5 15a5 5 0 0 0 5 5c7 0 11-7 10-16ZM5 20c3-5 6-8 11-11"/></g></svg>')
    (DOCS/'404.html').write_text(f'<!doctype html><meta charset="utf-8"><title>BambooX — Page not found</title><h1>Page not found</h1><p><a href="{BASE}/">Return to BambooX Field Lab</a></p>')
    print('Rendered all eight public pages with the browser AI adapter.')

if __name__=='__main__':build()
