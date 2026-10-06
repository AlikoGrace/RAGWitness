"""Replay prepared artifacts via official OTLP/HTTP JSON; no model calls.
Credentials read from --credentials, never included in exports or receipts.
"""
import argparse,base64,datetime,hashlib,json,re,secrets,time,urllib.request,urllib.parse
from pathlib import Path

def main():
 a=argparse.ArgumentParser();a.add_argument('--credentials',type=Path,required=True);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--export-only',action='store_true');args=a.parse_args()
 env=dict(re.findall(r'(LANGFUSE_\w+)="([^"]+)"',args.credentials.read_text()))
 auth=base64.b64encode((env['LANGFUSE_PUBLIC_KEY']+':'+env['LANGFUSE_SECRET_KEY']).encode()).decode();host=env['LANGFUSE_BASE_URL']
 def request(path,data=None):
  headers={'Authorization':'Basic '+auth,'Content-Type':'application/json','x-langfuse-ingestion-version':'4'}
  req=urllib.request.Request(host+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
  with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
 args.out.mkdir(parents=True,exist_ok=True);receipt=args.out/'receipt.json'
 if not args.export_only:
  projects=request('/api/public/projects')['data'];assert len(projects)==1 and projects[0]['name']=='RAGWitness-comparison'
  started=datetime.datetime.now(datetime.timezone.utc);spans=[];traces=[]
  def span(trace,name,payload,typ,parent=None):
   sid=secrets.token_hex(8);n=time.time_ns();attrs={'langfuse.observation.type':typ,'langfuse.observation.input':json.dumps(payload.get('input')),'langfuse.observation.output':json.dumps(payload.get('output')),'langfuse.trace.name':trace['name'],'langfuse.environment':'replication'}
   if 'model' in payload:attrs['langfuse.observation.model.name']=payload['model']
   if 'model_parameters' in payload:attrs['langfuse.observation.model.parameters']=json.dumps(payload['model_parameters'])
   s={'traceId':trace['trace_id'],'spanId':sid,'name':name,'kind':1,'startTimeUnixNano':str(n),'endTimeUnixNano':str(n),'attributes':[{'key':k,'value':{'stringValue':v}} for k,v in attrs.items()]}
   if parent:s['parentSpanId']=parent
   spans.append(s);return sid
  for entry in json.loads((args.prepared/'manifest.json').read_text())['artifacts']:
   p=args.prepared/entry['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'];d=json.loads(p.read_text())
   t={'case':entry['case'],'configuration':entry['configuration'],'trace_id':secrets.token_hex(16),'name':entry['case']+'-'+entry['configuration'],'payload_file':entry['file']};traces.append(t)
   root=span(t,t['name'],d if entry['configuration']=='bare' else d['root'],'span')
   if entry['configuration']!='bare':
    span(t,'retrieval',d['retrieval'],'retriever',root);span(t,'generation',d['generation'],'generation',root)
  payload={'resourceSpans':[{'resource':{'attributes':[{'key':'service.name','value':{'stringValue':'ragwitness-saved-artifact-replication'}}]},'scopeSpans':[{'scope':{'name':'ragwitness-replay','version':'1'},'spans':spans}]}]}
  (args.out/'otlp_request.json').write_text(json.dumps(payload,indent=2));response=request('/api/public/otel/v1/traces',payload)
  result={'project':projects[0]['name'],'project_id':projects[0]['id'],'host':host,'region':'EU','server_ui_version':'4.51.0','transport':'OTLP/HTTP JSON (no SDK)','model_calls':0,'synthetic_zero_durations':True,'from':(started-datetime.timedelta(minutes=1)).isoformat(),'to':(started+datetime.timedelta(hours=1)).isoformat(),'traces':traces,'span_count':len(spans),'ingestion_response':response}
  receipt.write_text(json.dumps(result,indent=2));print('Ingested',len(traces),'traces;',len(spans),'observations. Response:',response)
 else:result=json.loads(receipt.read_text())
 rows=[];cursor=None
 while True:
  params={'fromStartTime':result['from'],'toStartTime':result['to'],'fields':'core,basic,time,io,model,trace_context','limit':1000}
  if cursor:params['cursor']=cursor
  response=request('/api/public/v2/observations?'+urllib.parse.urlencode(params));rows.extend(response['data']);cursor=response.get('meta',{}).get('nextCursor')
  if not cursor:break
 (args.out/'raw_observations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 print('Exported',len(rows),'observations; expected',result['span_count'])
if __name__=='__main__':main()
