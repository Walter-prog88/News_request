"""Validate the published aggregate extracts and selected video identities."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = json.loads((ROOT / 'evidence/selected_results.json').read_text())
    for group in evidence['groups']:
        cases = group['cases']
        assert len(cases) == group['planned_rollouts']
        assert len({c['case_id'] for c in cases}) == len(cases)
        assert sum(c['completed'] for c in cases) == group['completed_rollouts']
        assert sum(c['success'] is True for c in cases) == group['task_successes']
    assets = json.loads((ROOT / 'evidence/video_manifest.json').read_text())
    for item in assets['videos']:
        path = ROOT / item['path']
        assert path.stat().st_size == item['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        assert item['expert_prefix'] is False
    print('PASS: case counts, original task outcomes, video sizes and SHA-256 match the published extracts.')


if __name__ == '__main__':
    main()
