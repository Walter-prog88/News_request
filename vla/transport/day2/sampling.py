#!/usr/bin/env python3
"""Frozen frame-index schedules with checkpoint-step-addressed Day2 resume."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shuffled_stream(pool, count, seed):
    """Uniform without replacement within each explicit pass through the pool."""
    pool = np.asarray(pool, dtype=np.int64)
    if pool.ndim != 1 or not len(pool) or len(np.unique(pool)) != len(pool):
        raise ValueError('Require a nonempty pool of distinct dataset indices')
    rng = np.random.default_rng(seed)
    values, rounds = [], []
    remaining, cycle = count, 0
    while remaining:
        permutation = rng.permutation(pool)
        take = min(remaining, len(pool))
        values.append(permutation[:take])
        rounds.append(np.full(take, cycle, dtype=np.int64))
        remaining -= take
        cycle += 1
    return np.concatenate(values), np.concatenate(rounds)


def validate_mapping(mapping, total_frames, expected_episodes=200):
    if len(mapping) != expected_episodes:
        raise ValueError('Unexpected train episode count')
    offset, seen = 0, set()
    for i, row in enumerate(mapping):
        if row['episode_index'] != i or row['from'] != offset or row['to'] != offset + row['length'] or row['length'] < 1:
            raise ValueError('Broken contiguous episode/frame mapping')
        if row['source_episode'] in seen:
            raise ValueError('Duplicate source episode')
        seen.add(row['source_episode'])
        offset = row['to']
    if offset != total_frames:
        raise ValueError('Frame coverage mismatch')


def make_arrays(total_frames, focus_indices, updates=5000, batch_size=4, seed=20261015):
    if updates < 1 or batch_size != 4:
        raise ValueError('Day2 comparison requires batch size four and positive updates')
    focus_indices = np.asarray(focus_indices, dtype=np.int64)
    if not len(focus_indices) or focus_indices.min() < 0 or focus_indices.max() >= total_frames:
        raise ValueError('Phase window indices are empty or outside train dataset')
    uniform, uniform_round = shuffled_stream(np.arange(total_frames), updates * batch_size, seed)
    focus, focus_round = shuffled_stream(focus_indices, updates * 2, [seed, 1])
    uniform, uniform_round = uniform.reshape(updates, 4), uniform_round.reshape(updates, 4)
    weighted, weighted_round = uniform.copy(), uniform_round.copy()
    weighted[:, 2:] = focus.reshape(updates, 2)
    weighted_round[:, 2:] = focus_round.reshape(updates, 2)
    source = np.zeros((2, updates, 4), dtype=np.uint8)
    source[1, :, 2:] = 1
    return {'indices': np.stack([uniform, weighted]), 'sources': source,
            'stream_rounds': np.stack([uniform_round, weighted_round]),
            'focus_indices': focus_indices}


class FrozenPlan:
    """Position is derived from successful optimizer state, never worker prefetch."""
    BRANCHES = ('uniform', 'handover_weighted')

    def __init__(self, metadata_path, branch):
        self.path = Path(metadata_path)
        self.metadata = json.loads(self.path.read_text())
        if branch not in self.BRANCHES or self.metadata.get('frozen') is not True:
            raise ValueError('Choose a known branch of a frozen schedule')
        self.branch = branch
        self.branch_index = self.BRANCHES.index(branch)
        plan_path = self.path.parent / self.metadata['array_file']
        if sha(plan_path) != self.metadata['array_sha256']:
            raise ValueError('Frozen sampling plan SHA-256 mismatch')
        with np.load(plan_path, allow_pickle=False) as z:
            self.indices = z['indices'][self.branch_index].copy()
            self.sources = z['sources'][self.branch_index].copy()
            self.rounds = z['stream_rounds'][self.branch_index].copy()
        self.mapping = self.metadata['mapping']
        validate_mapping(self.mapping, self.metadata['total_frames'])
        expected = (self.metadata['additional_updates'], self.metadata['batch_size'])
        if self.indices.shape != expected or self.sources.shape != expected or self.rounds.shape != expected:
            raise ValueError('Frozen schedule shape mismatch')
        if self.indices.min() < 0 or self.indices.max() >= self.metadata['total_frames']:
            raise ValueError('Frozen schedule addresses non-train frames')
        self.ends = np.asarray([row['to'] for row in self.mapping])

    def row(self, completed_updates):
        local_update = completed_updates - self.metadata['starting_completed_updates']
        if not 0 <= local_update < len(self.indices):
            raise ValueError('Completed update is outside this frozen schedule')
        return self.indices[local_update].copy()

    def sample_records(self, completed_updates):
        local_update = completed_updates - self.metadata['starting_completed_updates']
        indices = self.row(completed_updates)
        result = []
        for slot, index in enumerate(indices):
            row = self.mapping[int(np.searchsorted(self.ends, index, side='right'))]
            result.append({'slot': slot, 'dataset_index': int(index), 'episode_index': row['episode_index'],
                           'source_episode': row['source_episode'], 'frame_index': int(index - row['from']),
                           'sampling_source': 'handover_window' if self.sources[local_update, slot] else 'uniform_base',
                           'stream_round': int(self.rounds[local_update, slot])})
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase-labels', required=True, type=Path)
    p.add_argument('--m6000-run-config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', default=20261015, type=int)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('A frozen plan cannot be overwritten')
    labels = json.loads(args.phase_labels.read_text())
    if labels.get('source_split') != 'train' or labels.get('approved_for_weighting') is not True or labels.get('complete') is not True:
        raise ValueError('Require explicitly approved train-only phase labels')
    if labels.get('method') != 'physical_action_replay_finger_contacts_v2':
        raise ValueError('This plan only accepts the audited physical event label method')
    if not (labels.get('source_integrity') or {}).get('hdf5_sha256_verified_this_run'):
        raise ValueError('Phase labels must be bound to the fully verified original HDF5 subset')
    mapping = labels['mapping']
    total = labels['total_frames']
    validate_mapping(mapping, total)
    focus = np.zeros(total, dtype=bool)
    known = {row['source_episode']: row for row in mapping}
    for event in labels['episodes']:
        if not event.get('valid_handover_window'):
            continue
        row = known[event['source_episode']]
        start, stop = event['focus_frame_interval_exclusive_end']
        if not 0 <= start < stop <= row['length']:
            raise ValueError('Handover interval leaves its train episode')
        focus[row['from'] + start:row['from'] + stop] = True
    arrays = make_arrays(total, np.flatnonzero(focus), seed=args.seed)
    args.output.mkdir(parents=True)
    array_path = args.output / 'sampling_plan.npz'
    np.savez_compressed(array_path, **arrays)
    starts = np.asarray([row['from'] for row in mapping])
    summary = {}
    for branch_id, branch in enumerate(FrozenPlan.BRANCHES):
        values = arrays['indices'][branch_id]
        summary[branch] = {'drawn_frames': int(values.size), 'unique_frame_indices': int(len(np.unique(values))),
                           'unique_episode_indices': int(len(np.unique(np.searchsorted(starts, values, side='right') - 1))),
                           'actual_focus_frames': int(focus[values].sum()),
                           'intended_uniform_draws': int((arrays['sources'][branch_id] == 0).sum()),
                           'intended_focus_draws': int((arrays['sources'][branch_id] == 1).sum()),
                           'within_batch_duplicate_rows': int(sum(len(np.unique(row)) != len(row) for row in values)),
                           'maximum_stream_round': int(arrays['stream_rounds'][branch_id].max())}
    metadata = {'schema_version': 1, 'frozen': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'array_file': array_path.name, 'array_sha256': sha(array_path), 'numpy_version': np.__version__,
                'phase_labels_sha256': sha(args.phase_labels),
                'source_integrity': labels['source_integrity'],
                'source_m6000_run_config_sha256': sha(args.m6000_run_config),
                'source_split': 'train', 'total_frames': total, 'mapping': mapping,
                'source_conversion_manifest_sha256': labels['conversion_manifest_sha256'],
                'source_mapping_sha256': labels['mapping_sha256'],
                'starting_completed_updates': 6000, 'additional_updates': 5000, 'ending_completed_updates': 11000,
                'batch_size': 4, 'action_horizon': 10, 'model_seed': 42, 'sampling_seed': args.seed,
                'branches': list(FrozenPlan.BRANCHES), 'focus_pool_frames': int(focus.sum()), 'coverage': summary,
                'rule': 'Uniform frames are shuffled without replacement. Weighted keeps slots 0–1 of the same uniform plan and replaces slots 2–3 with an independent shuffled phase-window stream. A stream is reshuffled only after its full pool is exhausted. Repeated focus exposure is intentional.',
                'resume_rule': 'Use schedule row global_completed_updates−6000. Only successful optimizer updates advance the committed sample ledger; worker prefetch does not determine resume position.',
                'day1_boundary': 'This is a new Day2 schedule. Day1 restarted its seeded loader per invocation and did not save a sample ledger; this schedule does not pretend to reconstruct Day1 sample coverage.'}
    (args.output / 'sampling_plan.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'array_sha256': metadata['array_sha256'], 'coverage': summary}))


if __name__ == '__main__':
    main()
