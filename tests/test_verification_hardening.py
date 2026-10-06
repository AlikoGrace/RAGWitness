import hashlib
import json
import os
import ssl
from pathlib import Path

import pytest

from src.run_manager import RunManager
from src.timestamp_anchor import verify_timestamp_token

ROOT = Path(os.environ.get('RAGWITNESS_EVIDENCE_ROOT', str(Path(__file__).resolve().parents[1])))
CASE = ROOT / 'runs/20260512_152656_rag_L5_D4_ad2ccd'
TRUST = Path(os.environ.get('RAGWITNESS_TSA_CA_FILE', ssl.get_default_verify_paths().cafile or ''))
POLICY = '2.16.840.1.114412.7.1'
requires_evidence = pytest.mark.skipif(not (CASE / 'events.jsonl').is_file(), reason='Retained reference case is not installed')


@requires_evidence
def test_real_timestamp_requires_matching_bytes():
    token = (CASE / 'artifacts/rfc3161_timestamp.tsr').read_bytes()
    data = (CASE / 'events.jsonl').read_bytes()
    good = verify_timestamp_token(token, data, ca_file=TRUST, policy_oid=POLICY)
    assert good['ok'] and good['signature_verified'] and good['message_imprint_matched']
    assert not verify_timestamp_token(token, data + b'changed', ca_file=TRUST, policy_oid=POLICY)['ok']


@requires_evidence
def test_policy_and_trust_are_required(tmp_path):
    token = (CASE / 'artifacts/rfc3161_timestamp.tsr').read_bytes()
    data = (CASE / 'events.jsonl').read_bytes()
    assert not verify_timestamp_token(token, data, ca_file=TRUST, policy_oid='1.2.3.4')['ok']
    assert not verify_timestamp_token(token, data, ca_file=tmp_path / 'missing.pem')['ok']
    empty = tmp_path / 'empty.pem'
    empty.write_text('')
    assert not verify_timestamp_token(token, data, ca_file=empty)['ok']
    assert not verify_timestamp_token(b'not a timestamp', data, ca_file=TRUST)['ok']


@requires_evidence
def test_signature_corruption_is_rejected():
    token = bytearray((CASE / 'artifacts/rfc3161_timestamp.tsr').read_bytes())
    token[-1] ^= 1
    assert not verify_timestamp_token(bytes(token), (CASE / 'events.jsonl').read_bytes(), ca_file=TRUST, policy_oid=POLICY)['ok']


@requires_evidence
def test_empty_and_truncated_logs_are_rejected(tmp_path):
    p = tmp_path / 'events.jsonl'
    p.write_text('')
    assert not RunManager.verify_hash_chain(p)['ok']
    lines = (CASE / 'events.jsonl').read_text().splitlines()
    p.write_text('\n'.join(lines[:-1]) + '\n')
    assert not RunManager.verify_hash_chain(p)['ok']
    # Deliberately checking a live, incomplete chain requires opting out.
    assert RunManager.verify_hash_chain(p, require_finalized=False)['ok']


@requires_evidence
def test_expected_identity_rejects_regenerated_log(tmp_path):
    lines = [json.loads(l) for l in (CASE / 'events.jsonl').read_text().splitlines()]
    expected = lines[-1]['event_hash']
    lines[2]['payload']['query'] = 'changed evidence'
    previous = '0' * 64
    for e in lines:
        e['prev_hash'] = previous
        e.pop('event_hash')
        e['event_hash'] = hashlib.sha256(json.dumps(e, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        previous = e['event_hash']
    p = tmp_path / 'events.jsonl'
    p.write_text(''.join(json.dumps(e) + '\n' for e in lines))
    assert RunManager.verify_hash_chain(p)['ok']
    assert not RunManager.verify_hash_chain(p, expected_final_hash=expected)['ok']


@requires_evidence
def test_all_canonical_chains_still_pass():
    rows = json.loads((ROOT / 'analysis/tables/fresh_90run_metrics.json').read_text())
    assert len(rows) == 90
    for row in rows:
        assert RunManager.verify_hash_chain(ROOT / row['run_dir'] / 'events.jsonl')['ok'], row['run_id']


def _local_tsa_token(tmp_path, *, expired=False, signing_usage='timestamp'):
    """Create test-only trust/token material locally; no TSA/network calls."""
    import datetime
    import subprocess
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID

    now = datetime.datetime.now(datetime.timezone.utc)
    root_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    tsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'RAGWitness test-only CA')])
    ca = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
          .public_key(root_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - datetime.timedelta(days=60)).not_valid_after(now + datetime.timedelta(days=60))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), critical=False)
          .sign(root_key, hashes.SHA256()))
    usage = ExtendedKeyUsageOID.TIME_STAMPING if signing_usage == 'timestamp' else ExtendedKeyUsageOID.CODE_SIGNING
    tsa = (x509.CertificateBuilder()
           .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'RAGWitness test-only TSA')]))
           .issuer_name(name).public_key(tsa_key.public_key()).serial_number(x509.random_serial_number())
           .not_valid_before(now - datetime.timedelta(days=30))
           .not_valid_after(now - datetime.timedelta(days=1) if expired else now + datetime.timedelta(days=30))
           .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
           .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
           .add_extension(x509.ExtendedKeyUsage([usage]), critical=True)
           .add_extension(x509.SubjectKeyIdentifier.from_public_key(tsa_key.public_key()), critical=False)
           .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
           .sign(root_key, hashes.SHA256()))
    (tmp_path / 'root.pem').write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    (tmp_path / 'tsa.pem').write_bytes(tsa.public_bytes(serialization.Encoding.PEM))
    (tmp_path / 'tsa.key').write_bytes(tsa_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    (tmp_path / 'serial').write_text('01')
    (tmp_path / 'data').write_bytes(b'local verifier test evidence')
    conf = f'''[tsa]
default_tsa = tsa_config
[tsa_config]
serial = {tmp_path / 'serial'}
crypto_device = builtin
signer_cert = {tmp_path / 'tsa.pem'}
certs = {tmp_path / 'root.pem'}
signer_key = {tmp_path / 'tsa.key'}
signer_digest = sha256
default_policy = 1.2.3.4.5
digests = sha256
accuracy = secs:1
ordering = yes
tsa_name = yes
ess_cert_id_chain = yes
ess_cert_id_alg = sha256
'''
    (tmp_path / 'tsa.cnf').write_text(conf)
    def run(args):
        return subprocess.run(['openssl', *args], capture_output=True, text=True, timeout=15)
    query = run(['ts', '-query', '-data', str(tmp_path / 'data'), '-sha256', '-cert', '-out', str(tmp_path / 'query')])
    assert query.returncode == 0, query.stderr
    reply = run(['ts', '-reply', '-config', str(tmp_path / 'tsa.cnf'), '-queryfile', str(tmp_path / 'query'), '-token_out', '-out', str(tmp_path / 'token')])
    return reply, tmp_path / 'root.pem', (tmp_path / 'token').read_bytes() if (tmp_path / 'token').exists() else b''


def test_expired_signer_and_untrusted_root_are_rejected(tmp_path):
    valid = tmp_path / 'valid'
    valid.mkdir()
    reply, root, token = _local_tsa_token(valid)
    assert reply.returncode == 0, reply.stderr
    data = b'local verifier test evidence'
    assert verify_timestamp_token(token, data, ca_file=root, policy_oid='1.2.3.4.5')['ok']
    assert not verify_timestamp_token(token, data, ca_file=TRUST, policy_oid='1.2.3.4.5')['ok']
    expired = tmp_path / 'expired'
    expired.mkdir()
    reply, root, token = _local_tsa_token(expired, expired=True)
    assert reply.returncode == 0, reply.stderr
    assert not verify_timestamp_token(token, data, ca_file=root, policy_oid='1.2.3.4.5')['ok']


def test_non_timestamp_signer_cannot_issue_test_token(tmp_path):
    reply, root, token = _local_tsa_token(tmp_path, signing_usage='code')
    # OpenSSL TSA refuses certificates without the required exclusive EKU.
    assert reply.returncode != 0 or not verify_timestamp_token(token, b'local verifier test evidence', ca_file=root, policy_oid='1.2.3.4.5')['ok']


def test_ft_does_not_trust_false_cached_verification(tmp_path):
    from src.experiments.experiment_stealthy import _score_ft
    (tmp_path / 'events.jsonl').write_text('')
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts/integrity_verification.json').write_text(json.dumps({
        'verification': {'ok': False},
        'integrity': {'hash_chain': 'descriptive string, not proof'},
    }))
    result = _score_ft(tmp_path, {'attack_type': 'direct'})
    assert result['signals']['hash_chain_ok'] is False
    assert result['ft_score'] == 0.0
