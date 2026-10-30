"""Explicit runtime-only validation I/O override for the pinned official trainer."""
from pathlib import Path
import hashlib

def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def validation_loader_factory(original,config,events):
    """Change only old-valid workers1 ->4; preserve all other constructor args."""
    def construct(*args,**kwargs):
        dataset=kwargs.get('dataset',args[0] if args else None)
        mask=getattr(dataset,'filter_by_attribute',None)
        before=kwargs.get('num_workers',0)
        updated=dict(kwargs)
        if mask==config['train']['hdf5_validation_filter_key']:
            if not config['experiment']['validate'] or mask!='valid' or before!=1:
                raise ValueError('Unexpected official validation loader contract')
            updated['num_workers']=4
        elif mask!='train':
            raise ValueError('Unexpected dataset mask in controlled trainer')
        events.append({'mask':mask,'workers_before':before,'workers_after':updated.get('num_workers',0),
                       'batch_size':updated.get('batch_size'),'drop_last':updated.get('drop_last'),
                       'shuffle':updated.get('shuffle'),'sampler_class':type(updated.get('sampler')).__name__})
        return original(*args,**updated)
    return construct
