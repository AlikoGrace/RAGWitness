"""Verify retained timestamp bytes against current event files, without TSA calls."""
from pathlib import Path
import base64,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.timestamp_anchor import verify_timestamp_token
rows=json.loads((ROOT/'analysis/tables/fresh_90run_metrics.json').read_text());result=[]
for row in rows:
 if row['observability_level']!=5:continue
 d=ROOT/row['run_dir'];p=d/'artifacts/rfc3161_timestamp.tsr';integrity=json.loads((d/'integrity.json').read_text());embedded=integrity.get('rfc3161_anchor',{}).get('tsr_b64')
 token=p.read_bytes() if p.exists() else base64.b64decode(embedded) if embedded else None
 result.append({'run_id':row['run_id'],'present':token is not None,'verification':verify_timestamp_token(token,(d/'events.jsonl').read_bytes()) if token else None})
out=ROOT/'reproduced';out.mkdir(exist_ok=True);(out/'retained_anchors.json').write_text(json.dumps(result,indent=2))
present=sum(r['present'] for r in result);verified=sum(bool((r['verification'] or {}).get('ok')) for r in result)
print(f'Retained tokens: {present}/18; verified: {verified}/{present}; absent: {18-present}.')
if present!=10 or verified!=present:raise SystemExit('Verification differs from retained audit; inspect OpenSSL/trust/policy diagnostics, not saved success flags.')
