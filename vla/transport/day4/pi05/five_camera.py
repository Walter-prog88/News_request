"""Isolated five-independent-camera adapter; no upstream file/global mutation.

Pi0 already shares one image encoder across views. Clone only the two pinned
methods' function ASTs and add explicit image_keys to their preprocessing call.
The entire rest of each method, including sampling/RNG/loss, is unchanged.
"""
from __future__ import annotations
import ast
import copy
import dataclasses
import hashlib
import inspect
import textwrap

import flax.nnx as nnx
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi import transforms
from openpi.training import config as train_config
from transport_config import TransportDataConfig, make_config
from transport_policy import TransportSchema, TransportInputs, TransportOutputs, CAMERA_SLOTS

FIVE_CAMERA_KEYS = (*CAMERA_SLOTS, 'left_shoulder_0_rgb', 'right_shoulder_0_rgb')


def patched_method(original):
    original = inspect.unwrap(original)
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == '_model' and node.func.attr == 'preprocess_observation':
            if any(k.arg == 'image_keys' for k in node.keywords):
                raise ValueError('Pinned method already has image_keys; inspect upstream before adapting')
            node.keywords.append(ast.keyword(arg='image_keys', value=ast.Name(id='_DAY4_FIVE_CAMERA_KEYS', ctx=ast.Load())))
            found.append(node)
    if len(found) != 1:
        raise ValueError('Require exactly one preprocessing call in each pinned method')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace['_DAY4_FIVE_CAMERA_KEYS'] = FIVE_CAMERA_KEYS
    # A private function-global dictionary; upstream module globals are unchanged.
    exec(compile(tree, __file__+'::'+original.__name__, 'exec'), namespace)
    new = namespace[original.__name__]
    new.__day4_patch_audit__ = {
        'upstream_method_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'change': 'Exactly one added keyword: image_keys=FIVE_CAMERA_KEYS',
        'camera_keys': list(FIVE_CAMERA_KEYS), 'upstream_module_globals_mutated': False}
    return new


class FiveCameraPi0(Pi0):
    compute_loss = patched_method(Pi0.compute_loss)
    sample_actions = patched_method(Pi0.sample_actions)


@dataclasses.dataclass(frozen=True)
class FiveCameraConfig(Pi0Config):
    def create(self, rng):
        return FiveCameraPi0(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        observation, actions = super().inputs_spec(batch_size=batch_size)
        image = observation.images['base_0_rgb']
        mask = observation.image_masks['base_0_rgb']
        # Same bounded exception as upstream Pi0Config.inputs_spec: these are
        # abstract ShapeDtypeStruct specifications, not runtime image arrays.
        with at.disable_typechecking():
            return dataclasses.replace(observation,
                        images={key:image for key in FIVE_CAMERA_KEYS},
                        image_masks={key:mask for key in FIVE_CAMERA_KEYS}), actions


@dataclasses.dataclass(frozen=True)
class FiveCameraSchema(TransportSchema):
    def validate(self):
        if tuple(self.cameras) != FIVE_CAMERA_KEYS or len(set(self.cameras.values())) != 5:
            raise ValueError('Five distinct real camera keys in the fixed order are required; no composites')
        original_three = dataclasses.replace(self, cameras={key:self.cameras[key] for key in CAMERA_SLOTS})
        TransportSchema.validate(original_three)


@dataclasses.dataclass(frozen=True)
class FiveCameraInputs:
    schema: FiveCameraSchema

    def __call__(self, data):
        value = TransportInputs(self.schema)(data)
        value['image_mask'] = {key: True for key in FIVE_CAMERA_KEYS}
        if tuple(value['image']) != FIVE_CAMERA_KEYS:
            raise ValueError('Five-camera transformation lost or reordered an actual view')
        return value


@dataclasses.dataclass(frozen=True)
class FiveCameraDataConfig(TransportDataConfig):
    def create(self, assets_dirs, model_config):
        data = super().create(assets_dirs, model_config)
        return dataclasses.replace(data, data_transforms=transforms.Group(
            inputs=[FiveCameraInputs(self.schema)], outputs=[TransportOutputs()]))


def make_five_config(**kwargs):
    """Same model hyperparameters; five observation views, no new parameter heads."""
    schema = kwargs['schema']
    if not isinstance(schema, FiveCameraSchema):
        raise TypeError('Require explicitly audited FiveCameraSchema')
    original = make_config(**kwargs)
    return dataclasses.replace(original,
          model=FiveCameraConfig(**dataclasses.asdict(original.model)),
          data=FiveCameraDataConfig(repo_id=kwargs['repo_id'], schema=schema,
                                   base_config=train_config.DataConfig(prompt_from_task=True)),
          policy_metadata={**original.policy_metadata, 'camera_count':5, 'camera_keys':list(FIVE_CAMERA_KEYS),
                           'camera_composites':False, 'extension':'Shared vision encoder plus two additional independent prefix image sequences'})
