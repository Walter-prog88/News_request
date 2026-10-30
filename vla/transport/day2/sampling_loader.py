"""Runtime adapter for the frozen Day2 schedule, using the pinned openpi transforms."""
from __future__ import annotations
import multiprocessing
import numpy as np
import torch
import jax
from openpi import transforms
from openpi.models import model
from openpi.training import data_loader


class IndexedDataset:
    def __init__(self, raw, data_config, mapping):
        self.raw = raw
        self.mapping = mapping
        self.ends = np.asarray([row['to'] for row in mapping])
        self.transform = transforms.compose([
            *data_config.repack_transforms.inputs, *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs])

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        index = int(index)
        raw = self.raw[index]
        actual = np.asarray([int(np.asarray(raw[key]).item())
                             for key in ('index', 'episode_index', 'frame_index')], dtype=np.int64)
        row = self.mapping[int(np.searchsorted(self.ends, index, side='right'))]
        expected = np.asarray([index, row['episode_index'], index - row['from']])
        if not np.array_equal(actual, expected):
            raise ValueError(f'Actual LeRobot sample identity differs: {actual} vs {expected}')
        transformed = self.transform(raw)
        transformed['_day2_sample_identity'] = actual
        return transformed


class PlanSampler:
    def __init__(self, plan, completed_updates):
        local_update = completed_updates - plan.metadata['starting_completed_updates']
        plan.row(completed_updates)  # Bounds check without advancing mutable state.
        self.indices = plan.indices[local_update:].reshape(-1).copy()

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)


class PlannedLoader:
    def __init__(self, config, plan, completed_updates, sharding):
        if jax.process_count() != 1:
            raise ValueError('Day2 audited loader is single-process only')
        self.config = config.data.create(config.assets_dirs, config.model)
        if self.config.norm_stats is None:
            raise ValueError('Reuse the original train-only normalization; do not refit')
        raw = data_loader.create_torch_dataset(self.config, config.model.action_horizon, config.model)
        if len(raw) != plan.metadata['total_frames']:
            raise ValueError('Actual train dataset length differs from the frozen plan')
        dataset = IndexedDataset(raw, self.config, plan.mapping)
        generator = torch.Generator().manual_seed(plan.metadata['sampling_seed'])
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=PlanSampler(plan, completed_updates),
            shuffle=False, drop_last=False, num_workers=config.num_workers,
            multiprocessing_context=multiprocessing.get_context('spawn') if config.num_workers else None,
            persistent_workers=config.num_workers > 0, collate_fn=data_loader._collate_fn,
            worker_init_fn=data_loader._worker_init_fn, generator=generator)
        self.sharding = sharding

    def data_config(self):
        return self.config

    def __iter__(self):
        for batch in self.loader:
            identities = batch.pop('_day2_sample_identity')
            batch = jax.tree.map(lambda value: jax.make_array_from_process_local_data(self.sharding, value), batch)
            yield (model.Observation.from_dict(batch), batch['actions']), identities

    def close(self):
        # Torch has no public explicit persistent-worker close API. The iterator
        # cleanup is the same mechanism used when a DataLoader is collected.
        iterator = getattr(self.loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
