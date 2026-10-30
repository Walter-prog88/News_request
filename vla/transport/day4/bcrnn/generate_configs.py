"""Generate exact pinned official Transport configs, then explicit controlled variants.

Pure stdlib: extracts unmodified official function ASTs to avoid importing physics
when only creating JSON. No GPU, source dataset reads, or upstream file edits.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
PINS = {
    'dexmimicgen': '940e8a1b3ad70eb1925ada6b364b197de6bb2af9',
    'robomimic': '9ce065180dbcc38ddf8fa4218daa2e75b5fdc28a',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def frozen_sources(upstream):
    pins = json.loads((HERE/'source_pins.json').read_text())
    for name, expected in PINS.items():
        root = upstream/name
        if (root/'.git').exists():
            current = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
            if current != expected:
                raise ValueError('Wrong upstream commit: '+name)
        for rel, digest in pins['repositories'][name]['files_sha256'].items():
            if sha(root/rel) != digest:
                raise ValueError('Changed pinned upstream source: '+name+'/'+rel)
    return pins

def official_configs(upstream, dataset_dir, output_dir, dest):
    frozen_sources(upstream)
    dex, robo = upstream/'dexmimicgen', upstream/'robomimic'
    util = load_file('official_config_utils',dex/'dexmimicgen/utils/config_utils.py')
    hyper = load_file('official_hyperparam_utils',robo/'robomimic/utils/hyperparam_utils.py')
    source = dex/'scripts/generate_training_config.py'
    tree = ast.parse(source.read_text())
    funcs = ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)],type_ignores=[])
    namespace = {'os':os,'ConfigUtils':util,'ConfigGenerator':hyper.ConfigGenerator}
    exec(compile(funcs,str(source),'exec'),namespace)
    gens = namespace['make_generators'](str(robo/'robomimic/exps/templates/bc.json'),str(dataset_dir),str(output_dir))
    transport = [g for g in gens if g.parameters['train.data'].values == [str(dataset_dir/'generated/two_arm_transport.hdf5')]]
    if len(transport) != 1:
        raise ValueError('Cannot identify one official Transport generator')
    g = transport[0]
    g.generated_config_dir = str(dest)
    paths = g.generate()
    if len(paths) != 3:
        raise ValueError('Expected official three seeds')
    return [json.loads(Path(p).read_text()) for p in paths]

def differences(before,after,prefix=''):
    out = []
    if isinstance(before,dict) and isinstance(after,dict):
        for key in sorted(set(before)|set(after)):
            out += differences(before.get(key),after.get(key),prefix+'.'+key if prefix else key)
    elif before != after:
        out.append({'key':prefix,'official':before,'controlled':after})
    return out

def controlled_config(official, dataset, output, name, cameras, updates, epoch_steps, profile):
    if updates <= 0 or epoch_steps <= 0 or updates % epoch_steps:
        raise ValueError('Updates must be a positive multiple of epoch steps')
    cfg=copy.deepcopy(official)
    cfg['experiment']['name']=name
    cfg['train'].update(data=str(dataset),output_dir=str(output),hdf5_filter_key='train',
                        hdf5_validation_filter_key=None if profile else 'valid',num_epochs=updates//epoch_steps)
    cfg['experiment'].update(validate=not profile,keep_all_videos=True,epoch_every_n_steps=epoch_steps)
    cfg['experiment']['logging']['log_tb']=False
    # Independent collector handles exact frozen initial states and all videos.
    cfg['experiment']['rollout'].update(enabled=False,batched=False,num_batch_envs=1)
    cfg['experiment']['save'].update(every_n_epochs=1 if profile else 100,
                                     on_best_validation=False,on_best_rollout_success_rate=False)
    cfg['observation']['modalities']['obs']['rgb']=cfg['observation']['modalities']['obs']['rgb'][:cameras]
    return cfg

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=HERE/'upstream')
    p.add_argument('--dataset',type=Path,required=True,help='Existing 300-episode train/valid/test HDF5; opened later by preflight')
    p.add_argument('--official-dataset-dir',type=Path,required=True)
    p.add_argument('--training-output',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--updates',type=int,default=50000)
    p.add_argument('--epoch-steps',type=int,default=500)
    p.add_argument('--seed',type=int,choices=[201,202,203],default=201)
    p.add_argument('--cameras',type=int,choices=[3,5],default=5)
    p.add_argument('--name',default='day4_bcrnn_200_5cam_seed201')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    official_dir=args.output_dir/'official_reference_only';official_dir.mkdir()
    refs=official_configs(args.upstream,args.official_dataset_dir,args.training_output/'official_reference_only',official_dir)
    ref=next(c for c in refs if c['train']['seed']==args.seed)
    controlled=controlled_config(ref,args.dataset,args.training_output,args.name,args.cameras,args.updates,args.epoch_steps,False)
    profile=controlled_config(ref,args.dataset,args.training_output,args.name+'_profile200',args.cameras,200,200,True)
    write_new(args.output_dir/'controlled.json',controlled)
    write_new(args.output_dir/'profile200.json',profile)
    report={'status':'prepared_not_executed','pins':PINS,'official_updates_per_seed':300000,
            'controlled_updates':args.updates,'seed':args.seed,'camera_count':args.cameras,
            'controlled_differences':differences(ref,controlled),'profile_differences':differences(ref,profile),
            'official_reference_only':'Unchanged upstream recipe has no HDF5 filter and trains all released episodes. Do not run against data containing our frozen holdout, or claim this 200-demo control reproduces paper numbers.',
            'split_boundary':'200 source-train only; old valid for development loss/selection; frozen Day4 test only through separate collector after model selection.',
            'observation_boundary':'Same five cameras and18Dstate can be paired with pi0.5; BC uses sequence training+LSTM history and every-step execution. This is input/data matching, not equal architecture/history/compute.',
            'config_sha256':{str(x.relative_to(args.output_dir)):sha(x) for x in args.output_dir.rglob('*.json')}}
    write_new(args.output_dir/'generation.json',report)
    print(json.dumps({'output':str(args.output_dir),'controlled_updates':args.updates,'official_reference_only':True}))

if __name__=='__main__':main()
