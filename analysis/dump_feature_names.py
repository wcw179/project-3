import pickle, json
from pathlib import Path
p = Path('artifacts/models/xgb_mfe/EURUSDm/model_components.pkl')
obj = pickle.load(open(p,'rb'))
fn = obj.get('feature_names') or obj.get('features')
print(len(fn) if fn is not None else 'None')
out_dir = Path('artifacts/analysis/EURUSDm')
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'feature_names.json').write_text(json.dumps(fn, indent=2))
print('Wrote to', out_dir / 'feature_names.json')

