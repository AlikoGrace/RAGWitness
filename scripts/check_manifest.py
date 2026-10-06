"""Detect changes relative to the locally prepared release manifest."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'MANIFEST.sha256.json').read_text())
for name,digest in manifest['files'].items():
 p=ROOT/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
  raise SystemExit('Manifest mismatch: '+name)
print('PASS:',len(manifest['files']),'release file hashes match. This is a current release baseline, not historical custody proof.')
