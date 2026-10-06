"""Offline retained-evidence checks; never invokes a model, index, or TSA."""
from pathlib import Path
import argparse,ast,csv,json,os,statistics,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import forensic_reconstruction as fr
from src.metrics import attribution_accuracy

def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--statistics',action='store_true');ap.add_argument('--figures',action='store_true');args=ap.parse_args()
 os.chdir(ROOT)
 out=ROOT/'reproduced';out.mkdir(exist_ok=True)
 fr._CLEAN_CORPUS_JSONL=ROOT/'data/reference/clean_chunk_ids.jsonl';fr._load_clean_chunk_ids.cache_clear()
 assert len(fr._load_clean_chunk_ids())==15482
 rows=read(ROOT/'analysis/tables/fresh_90run_metrics.json');assert len(rows)==90
 checks=[]
 for row in rows:
  d=ROOT/row['run_dir'];cfg=read(d/'config.json');rec=fr.reconstruct_run(d,write=False)
  observed={'evidence_completeness':sum(rec['questions_answered'].values())/7,'reconstruction_fidelity':sum(rec['required_artifacts'].values())/7,'attribution_accuracy':attribution_accuracy(cfg.get('attack_type'),rec['reconstructed_attack_type'],cfg.get('expected_malicious_chunk_id'),rec['attributed_chunk_id'],rec['attributed_source'])}
  assert all(abs(observed[k]-row[k])<1e-12 for k in observed),(row['run_id'],observed)
  assert rec['integrity']['ok'],row['run_id']
  checks.append({'run_id':row['run_id'],'scores_match':True,'chain_ok':True})
 # Load only the original scoring definitions, avoiding optional model dependencies.
 p=ROOT/'src/experiments/experiment_stealthy.py';tree=ast.parse(p.read_text())
 selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_score_asr','_score_ft','_read_answer'] or isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='ASR_KEYWORDS' or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ASR_KEYWORDS' for t in n.targets)]
 ns={'Path':Path,'Any':object,'json':json};exec(compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec'),ns)
 stealth=read(ROOT/'analysis/tables/stealthy_attack_report.json')['rows'];assert len(stealth)==30
 ft_rows=[]
 for row in stealth:
  d=ROOT/'runs_stealthy'/row['run_id'];score=ns['_score_asr'](row['attack_id'],ns['_read_answer'](d));ft=ns['_score_ft'](d,{'attack_type':row['attack_type']})
  assert int(score['attack_success'])==row['asr']
  assert score['matched_keywords']==row['asr_matched_keywords']
  assert ft['ft_score']==row['ft_score'] and ft['signals']==row['ft_signals']
  ft_rows.append({'run_id':row['run_id'],'asr':row['asr'],'ft_numerator':sum(v is True for v in ft['signals'].values()),'ft_denominator':len(ft['applicable_signals']),'ft_score':ft['ft_score']})
 l1=statistics.mean(r['storage_bytes'] for r in rows if r['observability_level']==1)
 summary=[]
 for level in range(1,6):
  sub=[r for r in rows if r['observability_level']==level]
  summary.append({'level':level,'n':len(sub),'EC':statistics.mean(r['evidence_completeness'] for r in sub),'AA_attacks':statistics.mean(r['attribution_accuracy'] for r in sub if r['attack_type']!='baseline'),'RF':statistics.mean(r['reconstruction_fidelity'] for r in sub),'mean_storage_KiB':statistics.mean(r['storage_bytes'] for r in sub)/1024,'storage_ratio_fixed_L1_mean':statistics.mean(r['storage_bytes'] for r in sub)/l1})
 with (out/'canonical_summary.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
 sensitivity=read(ROOT/'analysis/tables/sensitivity_report.json')['raw_rows'];delta=[]
 for embed in ['minilm','mpnet']:
  for k in [3,5,10]:
   subsets=[[r for r in sensitivity if r['embed_key']==embed and r['top_k']==k and r['level']==l and r['attack_type']!='baseline'] for l in [1,2]]
   assert [len(x) for x in subsets]==[4,4]
   d=statistics.mean(r['AA'] for r in subsets[1])-statistics.mean(r['AA'] for r in subsets[0]);assert d==.5
   delta.append({'embed':embed,'k':k,'attack_only_n':4,'L2_minus_L1_AA':d})
 report={'canonical_count':90,'canonical_checks':checks,'stealthy_count':30,'stealthy_checks':ft_rows,'clean_ID_count':15482,'fixed_L1_mean_bytes':l1,'summary':summary,'sensitivity':delta,'statistics':'not requested','scope':'Saved artifact reconstruction; hash chaining is not independent custody or RFC3161 verification. Storage uses retained measurements, not changed directory sizes.'}
 if args.statistics:
  from src.experiments.statistical_analysis import build_statistical_report
  actual=build_statistical_report(rows);saved=read(ROOT/'analysis/tables/statistical_report.json')
  assert actual==saved,'Statistical result differs; inspect dependency versions.'
  (out/'statistical_report.json').write_text(json.dumps(actual,indent=2));report['statistics']='exactly reproduced'
 if args.figures:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(9,3.5));levels=[r['level'] for r in summary]
  for metric in ['EC','AA_attacks','RF']:axes[0].plot(levels,[r[metric] for r in summary],marker='o',label=metric)
  axes[0].set(xlabel='Logging level',ylabel='Score',ylim=(0,1.05));axes[0].legend()
  axes[1].bar(levels,[r['storage_ratio_fixed_L1_mean'] for r in summary]);axes[1].set(xlabel='Logging level',ylabel='Storage / fixed L1 mean')
  fig.tight_layout();fig.savefig(out/'retained_result_overview.pdf');plt.close(fig)
 (out/'verification.json').write_text(json.dumps(report,indent=2))
 print('PASS: 90 EC/AA/RF reconstructions and chains; 30 original ASR/FT scores; six +0.50 attack-only sensitivity deltas.')
 print('Results:',out)
if __name__=='__main__':main()
