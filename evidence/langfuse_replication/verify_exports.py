"""Verify read-back fidelity and evaluate declared artifact coverage.
Uses RAGWitness's retained analyzer with a shared clean reference; no labels as inputs.
"""
import argparse,collections,hashlib,json,sys
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--corpus',type=Path,required=True);a=ap.parse_args()
 sys.path.insert(0,str(a.repo));from src import forensic_reconstruction as fr
 fr._CLEAN_CORPUS_JSONL=a.corpus;fr._load_clean_chunk_ids.cache_clear()
 r=json.loads((a.root/'cloud_evidence/receipt.json').read_text());raw=json.loads((a.root/'cloud_evidence/raw_observations.json').read_text());by=collections.defaultdict(list)
 for row in raw:
  for field in ('input','output','modelParameters'):
   if isinstance(row.get(field),str):
    try: row[field]=json.loads(row[field])
    except json.JSONDecodeError: pass
  row['modelParameters']=row.get('modelParameters') or {}
  if row['modelParameters'].get('seed')=='null':row['modelParameters']['seed']=None
  by[row['traceId']].append(row)
 checks=[];scores=[]
 for trace in r['traces']:
  d=json.loads((a.root/'prepared'/trace['payload_file']).read_text());rows=by[trace['trace_id']];cfg=trace['configuration'];root=next(x for x in rows if x['parentObservationId'] is None)
  expected=d if cfg=='bare' else d['root'];checks.append({'trace_id':trace['trace_id'],'field':'root input/output','ok':root['input']==expected['input'] and root['output']==expected['output']})
  topk=[];has_prompt=False;has_params=False;model=False
  if cfg!='bare':
   ret=next(x for x in rows if x['type']=='RETRIEVER');gen=next(x for x in rows if x['type']=='GENERATION')
   checks.extend([{'trace_id':trace['trace_id'],'field':'retrieval','ok':ret['input']==d['retrieval']['input'] and ret['output']==d['retrieval']['output']},{'trace_id':trace['trace_id'],'field':'generation','ok':gen['input']==d['generation']['input'] and gen['output']==d['generation']['output'] and gen['model']==d['generation']['model']},{'trace_id':trace['trace_id'],'field':'modelParameters','ok':gen['modelParameters']==d['generation'].get('model_parameters',{})}])
   for c in ret['output']['chunks']:
    row={'rank':c['rank'],'chunk_id':c['chunk_id'],'metadata':{'source_url':c['source_url']}}
    if 'text' in c and c['text'] is not None:row['text']=c['text']
    topk.append(row)
   has_prompt=all(k in gen['input'] for k in ('system','user'));model=bool(gen['model']);has_params=all(k in gen['modelParameters'] for k in ('temperature','top_p'))
  direct=fr._looks_suspicious(root['input']['query']);suspect=fr._find_suspicious_chunk(topk)
  assess=direct or suspect is not None or has_prompt or any('text' in c for c in topk)
  coverage={'Q1':bool(assess),'Q2':bool(assess),'Q3':bool(direct or suspect is not None or assess),'Q4':bool(topk),'Q5':has_prompt,'Q6':has_prompt and has_params,'Q7':False}
  scores.append({'case':trace['case'],'configuration':cfg,'trace_id':trace['trace_id'],'coverage':coverage,'model_name_available':model,'model_parameters_available':has_params,'source': 'user_query' if direct else 'retrieved_chunk' if suspect else None,'attributed_chunk_id':suspect.get('chunk_id') if suspect else None})
 assert len(raw)==r['span_count'];assert all(c['ok'] for c in checks),[c for c in checks if not c['ok']]
 summaries={}
 for cfg in ('bare','declared_spans','enhanced_spans'):
  rows=[s for s in scores if s['configuration']==cfg];summaries[cfg]={q:{'answered':sum(s['coverage'][q] for s in rows),'total':len(rows)} for q in rows[0]['coverage']}
 result={'status':'EXPORTED_AND_FIDELITY_VERIFIED','observations':len(raw),'traces':len(r['traces']),'fidelity_checks':checks,'scores':scores,'summary':summaries,'scoring_scope':'Shared analyzer + identical trusted clean corpus access, not logger-alone attack detection. Q6 strict full prompt + model/temperature/top_p availability. Model-only availability reported separately. Q7 no native chain/anchor submitted or exported; no backend tampering test performed.','clean_corpus_sha256':hashlib.sha256(a.corpus.read_bytes()).hexdigest(),'model_calls':0,'export_transformation':'Enhanced modelParameters.seed exported as string null; normalized to None only for semantic comparison. Raw exports remain unchanged.'}
 (a.root/'replication_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(summaries,indent=2));print('All',len(checks),'read-back checks passed.')
if __name__=='__main__':main()
