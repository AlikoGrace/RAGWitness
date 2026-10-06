"""Prepare two explicitly specified logger configurations from saved artifacts.
No model, retrieval, Langfuse connection, or ground-truth scoring is performed.
"""
import argparse, hashlib, json
from pathlib import Path

def prepare(repo, out):
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((repo / 'runs_langfuse').glob('*/events.jsonl'))
    if len(paths) != 6:
        raise ValueError(f'Expected six retained cases; found {len(paths)}')
    manifest, evaluator = [], []
    for i, p in enumerate(paths, 1):
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        def payload(event):
            hits = [r['payload'] for r in rows if r['event'] == event]
            if len(hits) != 1: raise ValueError(f'{p}: expected one {event}')
            return hits[0]
        query = payload('query.received')['query']
        retrieval = payload('retrieval.completed')
        prompt = payload('prompt.built')
        gen = payload('generation.completed')
        label = f'case-{i:02d}'
        # Explicit bare trace: query and answer only; no automatic instrumentation.
        default = {'name': label, 'input': {'query': query}, 'output': {'answer': gen['answer']}}
        # Explicit spans reproduce the declared narrow comparison configuration.
        # Preserve query-time identifiers/URLs; omit evaluator labels and poison flags.
        chunks = [{'rank': c.get('rank'), 'chunk_id': c.get('chunk_id'),
                   'source_url': (c.get('metadata') or {}).get('source_url')}
                  for c in retrieval['topk']]
        spans = {'root': default, 'retrieval': {'input': {'query': query},
                 'output': {'chunks': chunks}},
                 'generation': {'input': {'system': prompt['system_prompt'],
                 'user': prompt['user_prompt']}, 'model': gen['model'],
                 'output': {'answer': gen['answer']}}}
        # Stronger comparator, kept separate: all generation settings + retrieved text.
        enhanced = {'root': default, 'retrieval': {'input': {'query': query},
                    'output': {'chunks': [dict(c, text=raw.get('text'))
                        for c, raw in zip(chunks, retrieval['topk'])]}},
                    'generation': dict(spans['generation'], model_parameters={
                        k: gen.get(k) for k in ('temperature', 'top_p', 'seed')})}
        for name, artifact in [('bare', default), ('declared_spans', spans), ('enhanced_spans', enhanced)]:
            dest = out / f'{label}_{name}.json'
            dest.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
            manifest.append({'case': label, 'configuration': name, 'file': dest.name,
                'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                'source_events_sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        scenario = payload('scenario.started')
        evaluator.append({'case': label, 'source_run': p.parent.name,
                          'scenario': scenario['attack_id'], 'attack_type': scenario['attack_type']})
    (out/'manifest.json').write_text(json.dumps({'status': 'PREPARED_ONLY_NOT_INGESTED',
        'ground_truth_logged': False, 'model_calls': 0, 'artifacts': manifest}, indent=2))
    (out/'evaluator_only_mapping.json').write_text(json.dumps(evaluator, indent=2))
    print(f'Prepared {len(manifest)} payloads from six saved cases; no Langfuse results claimed.')
if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();prepare(args.repo,args.out)
