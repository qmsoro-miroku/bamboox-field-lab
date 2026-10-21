from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent.parent
for route in ['', 'app','app/hub','app/inspection','app/camera','app/records','app/labeling','app/settings']:
    html=(ROOT/'docs'/route/'index.html').read_text(encoding='utf-8')
    assert '<!doctype html>' in html and 'web-api.js' in html and '{{' not in html,route
    assert 'href="/app' not in html and 'src="/assets' not in html,route
assert 'window.location.href =' not in (ROOT/'docs/app/records/index.html').read_text(encoding='utf-8')
for model in ['species','growth']:
    assert (ROOT/f'docs/models/{model}.onnx').stat().st_size>1000000
data=json.loads((ROOT/'docs/data/dataset.json').read_text(encoding='utf-8'))
assert len(data['images'])==data['total_images']==7050
assert sum(data['counts'].values())==data['object_count']
for item in data['images']:
    assert (ROOT/'docs/data/images'/item['split']/item['filename']).is_file()
assert not (ROOT/'database').exists() and not (ROOT/'uploads').exists()
print('PASS public pages, models, subpath links, dataset files, and excluded private runtime data')
