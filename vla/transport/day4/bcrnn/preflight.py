"""Read-only CPU audit of pinned BC config, existing splits, five cameras and actions.

Never reads held-out test pixels or runs policy inference. --loader additionally
constructs the real upstream training DataLoader with CUDA disabled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from generate_configs import frozen_sources, sha, write_new, HERE

LOW_DIM=['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos',
         'robot1_eef_pos','robot1_eef_quat','robot1_gripper_qpos']
CAMERAS=['agentview_image','robot0_eye_in_hand_image','robot1_eye_in_hand_image',
         'shouldercamera0_image','shouldercamera1_image']
ACTION_KEYS=['action_dict/right_rel_pos','action_dict/right_rel_rot_axis_angle','action_dict/right_gripper',
             'action_dict/left_rel_pos','action_dict/left_rel_rot_axis_angle','action_dict/left_gripper']

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--subset-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=HERE/'upstream')
    p.add_argument('--loader',action='store_true')
    p.add_argument('--loader-batch-size',type=int,default=2)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    frozen_sources(args.upstream)
    import h5py
    import numpy as np
    cfg=json.loads(args.config.read_text()); manifest=json.loads(args.subset_manifest.read_text())
    dataset=Path(cfg['train']['data'])
    if cfg['train']['hdf5_filter_key']!='train':raise ValueError('Controlled preflight requires train mask')
    if cfg['train']['hdf5_validation_filter_key']!=('valid' if cfg['experiment']['validate'] else None):raise ValueError('Expected old development valid mask or disabled validation')
    if cfg['train']['action_keys']!=ACTION_KEYS:raise ValueError('Unexpected action order')
    if cfg['observation']['modalities']['obs']['low_dim']!=LOW_DIM:raise ValueError('Unexpected18Dstate')
    cameras=cfg['observation']['modalities']['obs']['rgb']
    if cameras not in [CAMERAS,CAMERAS[:3]]:raise ValueError('Unknown camera contract')
    digest=file_sha(dataset)
    if digest!=manifest['subset_sha256']:raise ValueError('Dataset differs from frozen subset')
    records=[]; total=0
    with h5py.File(dataset,'r') as src:
        splits={key:[v.decode() if isinstance(v,bytes) else str(v) for v in src['mask'][key][:]] for key in ('train','valid','test')}
        for key in splits:
            if splits[key]!=manifest['splits'][key]:raise ValueError('Mask changed: '+key)
        if len(splits['train'])!=200 or len(splits['valid'])!=50 or len(splits['test'])!=50:raise ValueError('Expected200/50/50')
        if any(set(splits[a])&set(splits[b]) for a,b in [('train','valid'),('train','test'),('valid','test')]):raise ValueError('Split overlap')
        metadata=json.loads(src['data'].attrs['env_args'])
        for name in splits['train']:
            g=src['data'][name]; raw=np.asarray(g['actions'],dtype=np.float32)
            combined=np.concatenate([np.asarray(g[key],dtype=np.float32) for key in ACTION_KEYS],axis=-1)
            if raw.shape[1]!=14 or not np.array_equal(raw,combined):raise ValueError('Action dict differs from original raw controls: '+name)
            if not np.isfinite(raw).all():raise ValueError('Nonfinite action')
            for key,size in zip(LOW_DIM,[3,4,2,3,4,2]):
                arr=np.asarray(g['obs'][key]);
                if arr.shape!=(len(raw),size) or not np.isfinite(arr).all():raise ValueError('State invalid '+name+'/'+key)
            shapes={key:list(g['obs'][key].shape) for key in cameras}
            if any(shape!=[len(raw),84,84,3] for shape in shapes.values()):raise ValueError('Expected original84px RGB')
            if any(g['obs'][key].dtype!=np.uint8 for key in cameras):raise ValueError('Expecteduint8RGB')
            # Pixel decoding only in train; all expected views must be real nonconstant images.
            spot={}
            for key in cameras:
                frame=np.asarray(g['obs'][key][0]);
                if frame.min()==frame.max():raise ValueError('Constant image '+name+'/'+key)
                spot[key]=hashlib.sha256(frame.tobytes()).hexdigest()
            records.append({'episode':name,'frames':len(raw),'action_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                            'camera_shapes':shapes,'initial_train_rgb_sha256':spot})
            total+=len(raw)
    if total!=82036:raise ValueError('Train frame count changed')
    report={'status':'passed','dataset':str(dataset),'dataset_sha256':digest,'config_sha256':sha(args.config),
            'subset_manifest_sha256':sha(args.subset_manifest),'train_episodes':200,'train_frames':total,
            'split_counts':{k:len(v) for k,v in splits.items()},'camera_keys':cameras,'low_dim_keys':LOW_DIM,
            'all_train_action_dict_equals_original_controls':True,'train_records':records,'env_metadata':metadata,
            'test_boundary':'Whole-file bytes are hashed for integrity. Only test mask names are parsed; no test RGB/action/physics datasets are decoded or used for inference.',
            'loader_checked':False}
    if args.loader:
        sys.path.insert(0,str(args.upstream/'robomimic'))
        import torch
        import torchvision
        import robomimic.utils.obs_utils as ObsUtils
        import robomimic.utils.train_utils as TrainUtils
        from robomimic.config import config_factory
        config=config_factory(cfg['algo_name'],dic=cfg)
        ObsUtils.initialize_obs_utils_with_config(config)
        sets,_=TrainUtils.load_data_for_training(config,obs_keys=config.all_obs_keys)
        training=sets['data']
        if set(training.demos)!=set(splits['train']):raise ValueError('Actual loader selected wrong episodes')
        batch=next(iter(torch.utils.data.DataLoader(training,batch_size=args.loader_batch_size,num_workers=0)))
        def shape(x):
            if isinstance(x,dict):return {k:shape(v) for k,v in x.items()}
            return {'shape':list(x.shape),'dtype':str(x.dtype)}
        report['loader_checked']=True
        report['actual_batch']=shape(batch)
        report['actual_loader_episodes']=list(training.demos)
        report['versions']={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
                            'numpy':np.__version__,'h5py':h5py.__version__}
        report['cuda_visible_devices']=os.environ['CUDA_VISIBLE_DEVICES']
        report['loader_batch_size']=args.loader_batch_size
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_new(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k in ('status','train_episodes','train_frames','camera_keys','loader_checked')}))

if __name__=='__main__':main()
