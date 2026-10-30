#!/usr/bin/env python3
"""Freeze paired Day3 schedules only after current-weighted-model prefix review."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'day2'))
from sampling import sha,shuffled_stream,validate_mapping

BRANCHES=('handover_control','grasp_handover')
SOURCE_NAMES=('uniform_base','handover_window','grasp_window')


def make_arrays(total_frames,handover_indices,grasp_indices,updates=5000,seed=20261016):
    pools=[np.arange(total_frames),np.asarray(handover_indices,dtype=np.int64),np.asarray(grasp_indices,dtype=np.int64)]
    if updates<1 or total_frames<1: raise ValueError('Positive training dimensions required')
    for pool in pools:
        if not len(pool) or pool.min()<0 or pool.max()>=total_frames or len(np.unique(pool))!=len(pool):
            raise ValueError('Sampling pool must contain distinct train-only indices')
    u,ur=shuffled_stream(pools[0],updates*2,seed)
    h,hr=shuffled_stream(pools[1],updates*2,[seed,1])
    g,gr=shuffled_stream(pools[2],updates,[seed,2])
    a=np.concatenate([u.reshape(-1,2),h.reshape(-1,2)],axis=1)
    ar=np.concatenate([ur.reshape(-1,2),hr.reshape(-1,2)],axis=1)
    b,br=a.copy(),ar.copy();b[:,3]=g;br[:,3]=gr
    sources=np.zeros((2,updates,4),dtype=np.uint8);sources[:,:,2:]=1;sources[1,:,3]=2
    return {'indices':np.stack([a,b]),'sources':sources,'stream_rounds':np.stack([ar,br]),
            'handover_indices':pools[1],'grasp_indices':pools[2]}


class FrozenPlan:
    BRANCHES=BRANCHES
    def __init__(self,metadata_path,branch):
        self.path=Path(metadata_path);self.metadata=json.loads(self.path.read_text())
        m=self.metadata
        if branch not in BRANCHES or m.get('frozen') is not True or m.get('source_split')!='train': raise ValueError('Require approved frozen Day3 train schedule')
        if not m.get('prefix_review_approval_sha256'): raise ValueError('Prefix review approval absent')
        self.branch=branch;self.branch_index=BRANCHES.index(branch)
        path=self.path.parent/m['array_file']
        if sha(path)!=m['array_sha256']: raise ValueError('Frozen arrays changed')
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['indices'][0,:,:3],z['indices'][1,:,:3]): raise ValueError('Paired slots0–2 diverged')
            self.indices=z['indices'][self.branch_index].copy();self.sources=z['sources'][self.branch_index].copy();self.rounds=z['stream_rounds'][self.branch_index].copy()
        self.mapping=m['mapping'];validate_mapping(self.mapping,m['total_frames'])
        shape=(m['additional_updates'],m['batch_size'])
        if any(v.shape!=shape for v in (self.indices,self.sources,self.rounds)) or self.indices.min()<0 or self.indices.max()>=m['total_frames']: raise ValueError('Invalid schedule shape/range')
        self.ends=np.asarray([r['to'] for r in self.mapping])
    def row(self,completed_updates):
        local=completed_updates-self.metadata['starting_completed_updates']
        if not 0<=local<len(self.indices): raise ValueError('Optimizer step outside schedule')
        return self.indices[local].copy()
    def sample_records(self,completed_updates):
        local=completed_updates-self.metadata['starting_completed_updates']
        result=[]
        for slot,index in enumerate(self.row(completed_updates)):
            row=self.mapping[int(np.searchsorted(self.ends,index,side='right'))]
            result.append({'slot':slot,'dataset_index':int(index),'episode_index':row['episode_index'],
                           'source_episode':row['source_episode'],'frame_index':int(index-row['from']),
                           'sampling_source':SOURCE_NAMES[int(self.sources[local,slot])],
                           'stream_round':int(self.rounds[local,slot])})
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--day2-plan',required=True,type=Path)
    p.add_argument('--grasp-labels',required=True,type=Path)
    p.add_argument('--source-checkpoint',required=True,type=Path)
    p.add_argument('--prefix-review-approval',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',type=int,default=20261016)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Frozen plan cannot be overwritten')
    old=json.loads(a.day2_plan.read_text());grasp=json.loads(a.grasp_labels.read_text());approval=json.loads(a.prefix_review_approval.read_text())
    source=a.source_checkpoint
    if source.name!='10999' or not(source/'params').is_dir() or not(source/'train_state').is_dir(): raise ValueError('Require full M11000 checkpoint')
    run=source/'transport_run.json';state=source/'day2_sampling_state.json'
    state_data=json.loads(state.read_text())
    if state_data.get('branch')!='handover_weighted' or state_data.get('global_completed_updates')!=11000 or state_data.get('plan_array_sha256')!=old['array_sha256']:
        raise ValueError('Wrong Day2 origin')
    required={'approved':True,'source_checkpoint_run_sha256':sha(run),'grasp_labels_sha256':sha(a.grasp_labels),
              'comparison':'A_50uniform_50handover__B_50uniform_25handover_25grasp'}
    if any(approval.get(k)!=v for k,v in required.items()): raise ValueError('Approval missing or not bound to current candidate')
    if not approval.get('prefix_diagnostic_evidence_sha256') or not approval.get('decision_reason'): raise ValueError('Root must record current-model prefix diagnostic review')
    if grasp.get('source_split')!='train' or grasp.get('complete') is not True or grasp.get('method')!='original_close_transition_with_replayed_grasp_lift_v1': raise ValueError('Bad grasp label provenance')
    if grasp['mapping']!=old['mapping'] or grasp['total_frames']!=old['total_frames'] or grasp['day2_phase_labels_sha256']!=old['phase_labels_sha256']: raise ValueError('Dataset mapping/physical labels changed')
    if grasp['dataset_sha256']!=old['source_integrity']['hdf5_sha256_verified_this_run']: raise ValueError('HDF source changed')
    old_arrays=a.day2_plan.parent/old['array_file']
    if sha(old_arrays)!=old['array_sha256']: raise ValueError('Old phase pool changed')
    with np.load(old_arrays,allow_pickle=False) as z: handover=z['focus_indices'].copy()
    focus=np.zeros(old['total_frames'],dtype=bool);transition=np.zeros_like(focus)
    if len(grasp['episodes'])!=200 or {r['source_episode'] for r in grasp['episodes']}!={r['source_episode'] for r in old['mapping']}: raise ValueError('Incomplete grasp episode coverage')
    for row in grasp['episodes']:
        if not row['valid_grasp_window']:continue
        start,stop=row['focus_frame_interval_exclusive_end']
        if not 0<=start<stop<=row['length']: raise ValueError('Grasp window outside train episode')
        focus[row['from']+start:row['from']+stop]=True
        for anchor in row['transition_target_anchor_frames']:
            if not start<=anchor<stop: raise ValueError('Transition anchor outside approved window')
            transition[row['from']+anchor]=True
    arrays=make_arrays(old['total_frames'],handover,np.flatnonzero(focus),seed=a.seed)
    summary={}
    for i,branch in enumerate(BRANCHES):
        v=arrays['indices'][i];s=arrays['sources'][i]
        summary[branch]={'drawn_frames':int(v.size),'unique_frame_indices':int(np.unique(v).size),
                         'intended_source_draws':{name:int((s==j).sum()) for j,name in enumerate(SOURCE_NAMES)},
                         'actual_grasp_window_draws':int(focus[v].sum()),
                         'actual_handover_window_draws':int(np.isin(v,handover).sum()),
                         'actual_close_transition_target_draws':int(transition[v].sum()),
                         'grasp_source_close_transition_target_draws':int((transition[v]&(s==2)).sum()),
                         'within_batch_duplicate_rows':int(sum(len(np.unique(r))<4 for r in v))}
    if summary[BRANCHES[1]]['grasp_source_close_transition_target_draws']==0: raise ValueError('New focus stream contains no real command transitions')
    a.output.mkdir(parents=True)
    array_path=a.output/'sampling_plan.npz';np.savez_compressed(array_path,**arrays)
    metadata={'schema_version':1,'frozen':True,'created_utc':datetime.now(timezone.utc).isoformat(),
              'source_split':'train','array_file':array_path.name,'array_sha256':sha(array_path),
              'numpy_version':np.__version__,'starting_completed_updates':11000,'additional_updates':5000,'ending_completed_updates':16000,
              'batch_size':4,'action_horizon':10,'model_seed':42,'sampling_seed':a.seed,'branches':list(BRANCHES),
              'mapping':old['mapping'],'total_frames':old['total_frames'],
              'source_conversion_manifest_sha256':old['source_conversion_manifest_sha256'],
              'source_mapping_sha256':old['source_mapping_sha256'],
              'source_m11000_run_config_sha256':sha(run),'source_m11000_day2_state_sha256':sha(state),
              'source_day2_plan_sha256':sha(a.day2_plan),'grasp_labels_sha256':sha(a.grasp_labels),
              'prefix_review_approval_sha256':sha(a.prefix_review_approval),'prefix_review':approval,
              'coverage':summary,'grasp_pool_frames':int(focus.sum()),'handover_pool_frames':len(handover),
              'transition_target_anchor_pool_frames':int(transition.sum()),
              'rule':'A slots0–1 uniform, slots2–3 old handover pool. B slots0–2 identical to A; only slot3 replaced by independently shuffled grasp-window stream. Explicit pool cycles; no replacement within each stream cycle. Pools can overlap.',
              'resume_rule':'next row = restored optimizer completed_updates−11000; only successful optimizer updates enter the ledger.',
              'boundary':'Study is a replacement of 25% handover exposure by validated grasp-window exposure, with resulting repetition/episode coverage changes reported; not a pure causal grasp-pose intervention and not off-policy corrective data.'}
    (a.output/'sampling_plan.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'coverage':summary,'array_sha256':metadata['array_sha256']}))
if __name__=='__main__':main()
