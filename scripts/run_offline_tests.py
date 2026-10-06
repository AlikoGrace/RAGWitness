"""Run the relevant RAGWitness regression tests without live TSA requests."""
import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
args = parser.parse_args()
repo = args.repo.resolve()
sys.path.insert(0, str(repo))
os.chdir(repo)
import src.timestamp_anchor as anchor
import pytest

def offline_request(*args, **kwargs):
    raise RuntimeError('Live TSA requests disabled for offline regression tests')

anchor.request_timestamp = offline_request
raise SystemExit(pytest.main([
    '-q', '-p', 'no:cacheprovider',
    'tests/test_verification_hardening.py',
    'tests/test_forensic_reconstruction.py',
    'tests/test_metrics.py',
    'tests/test_pipeline_logging_levels.py',
    'tests/test_observability_config.py',
]))
