#!/usr/bin/env python3
"""Frozen validation-anchor action imitation, distinct from closed-loop success.

CPU freeze/check/client only. This module never loads a model, imports a GPU
framework, starts/stops a server, or changes training / rollout code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from transport_policy import ACTION_DIM, CAMERA_SLOTS, TransportInputs, TransportSchema

FRAMES = (0, 100, 200)
HORIZON = 10
POSE = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
TRANSLATION = (0, 1, 2, 7, 8, 9)
ROTATION = (3, 4, 5, 10, 11, 12)
GRIPPER = (6, 13)
CLASS_NAMES = {-1: "open", 0: "hold", 1: "close"}


def now():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=json_default).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    return sha(Path(path).read_bytes())


def array_hash(value):
    return sha(np.ascontiguousarray(value).tobytes())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=json_default)
        stream.write("\n")


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def readonly_hdf5(path):
    import h5py  # Freeze / real-source audit only; synthetic CPU checks need no HDF5 install.
    return h5py.File(path,"r")


def frozen_payload(envelope):
    payload = envelope["anchor_manifest"]
    if sha(canonical(payload)) != envelope["anchor_manifest_sha256"]:
        raise ValueError("Frozen anchor manifest was changed.")
    if payload["frames"] != list(FRAMES) or payload["action_horizon"] != HORIZON or len(payload["anchors"]) != 15:
        raise ValueError("Require exactly first five validation episodes × frames 0/100/200 × horizon 10.")
    return payload


def source_contract(dataset, schema_path, conversion_path):
    schema = TransportSchema.load(schema_path)
    schema_json = read_json(schema_path)
    conversion = read_json(conversion_path)
    if schema.state_dim != 18 or schema.arm_order != ("robot0", "robot1"):
        raise ValueError("This diagnostic requires the audited 18-D dual-Panda state order.")
    if not conversion.get("complete") or conversion.get("action_dim") != 14 or conversion.get("state_dim") != 18:
        raise ValueError("Require a completed 14-action/18-state conversion audit.")
    if conversion.get("images_flipped") is not False or conversion.get("images_resized") is not False:
        raise ValueError("Read unchanged current HDF5 RGB; no runtime image flip belongs here.")
    if conversion.get("action_representation") != "relative_eef_controller_command":
        raise ValueError("Expected relative EEF controller inputs.")
    if not np.isclose(schema.fps, conversion["fps"]) or not np.isclose(schema.fps, 20):
        raise ValueError("The existing 20-Hz controller convention must be preserved.")
    state_key = schema_json.get("runtime", {}).get("combined_state_key", "observation.state")
    if {field.key for field in schema.state_fields} != {state_key}:
        raise ValueError("Require the existing combined 18-D state feature; do not guess another repack.")
    camera_bindings = {}
    for slot in CAMERA_SLOTS:
        feature = schema.cameras[slot]
        choices = [camera for camera in conversion["image_shapes_hwc"]
                   if feature == f"observation.images.{camera}"]
        if len(choices) != 1:
            raise ValueError(f"Cannot bind audited HDF5 camera: {feature}")
        camera_bindings[slot] = choices[0]
    config = conversion["env_metadata"]["env_kwargs"]["controller_configs"]
    arm_controller = config["body_parts"]["right"]
    if (config["type"] != "BASIC" or arm_controller["type"] != "OSC_POSE"
            or arm_controller["input_type"] != "delta" or arm_controller["input_ref_frame"] != "base"
            or arm_controller["input_min"] != -1 or arm_controller["input_max"] != 1):
        raise ValueError("Controller bounds or representation differ from the existing audited setup.")
    with readonly_hdf5(dataset) as source:
        if json.loads(decode(source["data"].attrs["env_args"])) != conversion["env_metadata"]:
            raise ValueError("HDF5 environment and conversion metadata differ.")
    return schema, schema_json, conversion, state_key, camera_bindings


def extract_anchor(group, frame, schema, conversion, state_key, camera_bindings):
    length = len(group["actions"])
    if length < 210 or frame + HORIZON > length:
        raise ValueError(f"Episode {group.name} has only {length} frames; anchors require >=210 with no clamping.")
    pieces = []
    for field in conversion["state_fields"]:
        value = np.asarray(group[field["hdf5_key"]][frame], dtype=np.float32)
        if value.shape != (field["dimension"],) or not np.isfinite(value).all():
            raise ValueError(f"Invalid current state: {field}")
        pieces.append(value)
    state = np.concatenate(pieces)
    target = np.asarray(group["actions"][frame:frame+HORIZON], dtype=np.float32)
    if state.shape != (18,) or target.shape != (10, 14) or not np.isfinite(target).all():
        raise ValueError("Current state / target action shape or finiteness mismatch.")
    observation = {state_key: state, schema.prompt_key: conversion["task"]}
    images = {}
    for slot in CAMERA_SLOTS:
        camera = camera_bindings[slot]
        image = np.asarray(group[f"obs/{camera}_image"][frame])
        if image.dtype != np.uint8 or image.shape != tuple(conversion["image_shapes_hwc"][camera]):
            raise ValueError(f"Invalid current RGB for {camera}")
        observation[schema.cameras[slot]] = np.ascontiguousarray(image)
        images[camera] = {"shape_hwc":list(image.shape), "pixel_sha256":array_hash(image)}
    parsed = TransportInputs(schema)(observation)
    if "actions" in parsed or parsed["state"].shape != (18,):
        raise AssertionError("Inference payload must contain current observations only.")
    record = {"episode_length":length, "state_float32":state.tolist(),
              "state_float32_sha256":array_hash(state), "current_images":images,
              "gt_actions_raw_float32":target.tolist(), "gt_actions_raw_float32_sha256":array_hash(target)}
    return observation, record


def freeze(args):
    if args.output.exists():
        raise FileExistsError("Frozen anchors already exist; reuse them without rewriting.")
    schema, schema_json, conversion, state_key, cameras = source_contract(args.dataset,args.schema,args.conversion_manifest)
    anchors = []
    with readonly_hdf5(args.dataset) as source:
        splits = {key:[decode(v) for v in source[f"mask/{key}"][:]] for key in ("train","valid","test")}
        if [len(splits[k]) for k in ("train","valid","test")] != [200,50,50]:
            raise ValueError("Require the frozen 200/50/50 subset.")
        if any(len(v)!=len(set(v)) for v in splits.values()) or any(
                set(splits[a])&set(splits[b]) for a,b in (("train","valid"),("train","test"),("valid","test"))):
            raise ValueError("Duplicate or overlapping episode masks.")
        selected = splits["valid"][:5]
        for local_episode,episode in enumerate(selected):
            group=source["data"][episode]
            for frame in FRAMES:
                _,record=extract_anchor(group,frame,schema,conversion,state_key,cameras)
                anchors.append({"anchor_id":f"{episode}_frame_{frame:04d}","source_episode":episode,
                                "validation_mask_index":local_episode,"frame_index_zero_based":frame,
                                "target_frame_indices":list(range(frame,frame+10)),
                                "noise_seed":int(args.seed)+len(anchors),**record})
    subset_path=args.dataset.with_suffix(".manifest.json")
    subset=read_json(subset_path)
    payload={"version":1,"created_at_utc":now(),"diagnostic":"held-out expert-observation open-loop action imitation",
             "claim_boundary":"Development diagnostic chosen after earlier closed-loop results; not a blind final test, success rate or Sim2Real.",
             "dataset":str(args.dataset.resolve()),"dataset_bytes":args.dataset.stat().st_size,
             "recorded_subset_sha256":subset["subset_sha256"],"full_hdf5_rehash_performed_here":False,
             "subset_manifest":str(subset_path.resolve()),"subset_manifest_sha256":file_hash(subset_path),
             "schema_path":str(args.schema.resolve()),"schema_sha256":file_hash(args.schema),"schema_file":schema_json,
             "conversion_manifest":str(args.conversion_manifest.resolve()),"conversion_manifest_sha256":file_hash(args.conversion_manifest),
             "norm_stats_path":str(args.norm_stats.resolve()),"norm_stats_sha256":file_hash(args.norm_stats),
             "repo_id":args.repo_id,"selected_validation_episodes":selected,
             "frames":list(FRAMES),"action_horizon":10,"state_dim":18,"real_action_dim":14,
             "constant_prompt":conversion["task"],"images_flipped":False,"future_images_or_states_provided":False,
             "pose_indices":list(POSE),"translation_indices":list(TRANSLATION),"rotation_indices":list(ROTATION),
             "gripper_indices":list(GRIPPER),"gripper_sign_rule":"strict numpy.sign: -1=open, 0=hold, +1=close; no epsilon threshold",
             "action_bounds":{"low":[-1.0]*14,"high":[1.0]*14},
             "noise_seed_rule":"base seed + anchor ordinal, explicitly reset for every anchor for every model",
             "anchors":anchors}
    envelope={"anchor_manifest":payload,"anchor_manifest_sha256":sha(canonical(payload))}
    write_json(args.output,envelope,exclusive=True)
    return {"status":"FROZEN","anchors":15,"manifest":str(args.output.resolve()),
            "anchor_manifest_sha256":envelope["anchor_manifest_sha256"],"predictions_run":0}


def gripper_metrics(prediction,target):
    p=np.sign(prediction).astype(int).reshape(-1)
    y=np.sign(target).astype(int).reshape(-1)
    counts={CLASS_NAMES[c]:int(np.sum(y==c)) for c in (-1,0,1)}
    recalls={CLASS_NAMES[c]:float(np.mean(p[y==c]==c)) if np.any(y==c) else None for c in (-1,0,1)}
    confusion=[[int(np.sum((y==a)&(p==b))) for b in (-1,0,1)] for a in (-1,0,1)]
    return {"commands":len(y),"sign_accuracy":float(np.mean(p==y)),"gt_class_counts":counts,
            "predicted_class_counts":{CLASS_NAMES[c]:int(np.sum(p==c)) for c in (-1,0,1)},
            "class_recall":recalls,"balanced_accuracy_over_present_gt_classes":float(np.mean([v for v in recalls.values() if v is not None])),
            "confusion_rows_gt_columns_pred_open_hold_close":confusion,
            "always_open_accuracy":counts["open"]/len(y),"always_close_accuracy":counts["close"]/len(y),
            "always_hold_accuracy":counts["hold"]/len(y)}


def action_metrics(prediction,target,low,high):
    prediction=np.asarray(prediction,dtype=np.float32)
    target=np.asarray(target,dtype=np.float32)
    if prediction.shape != target.shape or prediction.shape[-2:] != (10,14):
        raise ValueError(f"Expected matching (...,10,14), got {prediction.shape}/{target.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("All action values must be finite.")
    clipped=np.clip(prediction,low,high)
    gt_clipped=np.clip(target,low,high)
    output={}
    for name,pred,gt in (("raw",prediction,target),("clipped",clipped,gt_clipped)):
        error=np.abs(pred.astype(np.float64)-gt.astype(np.float64))
        output[name]={"pose12_mae":float(error[...,POSE].mean()),
                      "translation6_mae":float(error[...,TRANSLATION].mean()),
                      "rotation6_mae":float(error[...,ROTATION].mean()),
                      "first_action_pose12_mae":float(error[...,0,POSE].mean()),
                      "per_pose_dimension_mae":{str(i):float(error[...,i].mean()) for i in POSE}}
    output["gripper"]={"pooled":gripper_metrics(prediction[...,GRIPPER],target[...,GRIPPER]),
                       "robot0":gripper_metrics(prediction[...,6],target[...,6]),
                       "robot1":gripper_metrics(prediction[...,13],target[...,13]),
                       "raw_and_clipped_signs_identical":bool(np.array_equal(np.sign(prediction[...,GRIPPER]),np.sign(clipped[...,GRIPPER]))),
                       "within_window_gt_sign_transitions_by_arm":{f"robot{arm}":int(np.count_nonzero(np.diff(np.sign(target[...,column]),axis=-1)))
                                                                  for arm,column in enumerate(GRIPPER)}}
    output["prediction_clip_ratio"]={"pose12":float(np.mean((prediction[...,POSE]<low[list(POSE)])|(prediction[...,POSE]>high[list(POSE)]))),
                                     "gripper2":float(np.mean((prediction[...,GRIPPER]<low[list(GRIPPER)])|(prediction[...,GRIPPER]>high[list(GRIPPER)])))}
    output["gt_commands_outside_bounds"]=int(np.count_nonzero((target<low)|(target>high)))
    return output


def validate_metadata(metadata,schema,payload,expected_model,expected_checkpoint):
    expected=schema.summary()
    for key,value in expected.items():
        if key not in metadata or canonical(metadata[key])!=canonical(value):
            raise ValueError(f"Server/schema mismatch: {key}")
    if metadata.get("model")!=expected_model or not metadata.get("episode_seed_reset"):
        raise ValueError("Wrong model kind or no per-anchor noise reset support.")
    if Path(metadata.get("checkpoint","")).resolve()!=Path(expected_checkpoint).resolve():
        raise ValueError("Server checkpoint differs from the explicitly expected checkpoint.")
    if file_hash(payload["norm_stats_path"])!=payload["norm_stats_sha256"]:
        raise ValueError("The frozen common normalization file changed.")
    if expected_model=="lora":
        checkpoint=Path(expected_checkpoint)
        sidecar=read_json(checkpoint/"transport_run.json")
        if canonical(sidecar["schema_file"])!=canonical(payload["schema_file"]):
            raise ValueError("Checkpoint schema differs from frozen anchors.")
        if sidecar["arguments"]["repo_id"]!=payload["repo_id"] or sidecar["arguments"]["horizon"]!=10:
            raise ValueError("Checkpoint repository / action horizon mismatch.")
        if (sidecar["training_invariants"]["norm_stats_sha256"]!=payload["norm_stats_sha256"]
                or file_hash(checkpoint/"assets"/payload["repo_id"]/"norm_stats.json")!=payload["norm_stats_sha256"]):
            raise ValueError("Checkpoint normalization differs from frozen common stats.")
    return {"server_schema_and_checkpoint_matched":True,"common_norm_file_sha256_matched":True,
            "lora_checkpoint_norm_snapshot_matched":expected_model=="lora",
            "normalization_attestation_boundary":"Existing server metadata does not expose an in-memory norm hash; common file and LoRA checkpoint snapshots are checked, with the unchanged audited serving code."}


def check_idle():
    """Fail closed on this Linux host while training or main evaluation owns GPU."""
    proc=Path("/proc")
    if not proc.exists():
        raise RuntimeError("Live evaluation requires the Linux process-idle guard; CPU self-check works elsewhere.")
    blocking={"transport_train.py","post6000_pipeline.py","evaluate_stage.py","transport_rollout.py"}
    found=[]
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            parts=(item/"cmdline").read_bytes().split(b"\0")
            names={Path(part.decode(errors="replace")).name for part in parts if part}
            if names&blocking:
                found.append({"pid":int(item.name),"commands":sorted(names&blocking)})
        except (OSError,PermissionError):
            continue
    if found:
        raise RuntimeError(f"Training/main pipeline still running; no diagnostic predictions permitted: {found}")
    return {"linux_process_guard":"passed","blocking_script_names":sorted(blocking)}


class LocalWireClient:
    def __init__(self,port,timeout):
        import websockets.sync.client
        from openpi_client import msgpack_numpy
        self.codec=msgpack_numpy
        self.packer=msgpack_numpy.Packer()
        self.timeout=timeout
        self.connection=websockets.sync.client.connect(f"ws://127.0.0.1:{port}",compression=None,
                                                      max_size=None,open_timeout=10,close_timeout=5)
        self.metadata=msgpack_numpy.unpackb(self.connection.recv(timeout=30))

    def infer(self,observation):
        self.connection.send(self.packer.pack(observation))
        reply=self.connection.recv(timeout=self.timeout)
        if isinstance(reply,str):
            raise RuntimeError(reply)
        return self.codec.unpackb(reply)

    def close(self):
        self.connection.close()


def evaluate(args):
    idle=check_idle()  # No model calls while the main experiment is active.
    envelope=read_json(args.anchors)
    payload=frozen_payload(envelope)
    schema_path=Path(payload["schema_path"])
    conversion_path=Path(payload["conversion_manifest"])
    for path,key in ((schema_path,"schema_sha256"),(conversion_path,"conversion_manifest_sha256"),
                     (Path(payload["subset_manifest"]),"subset_manifest_sha256")):
        if file_hash(path)!=payload[key]:
            raise ValueError(f"Frozen source metadata changed: {path}")
    dataset=Path(payload["dataset"])
    if dataset.stat().st_size!=payload["dataset_bytes"]:
        raise ValueError("Frozen dataset file size changed.")
    schema,schema_json,conversion,state_key,cameras=source_contract(dataset,schema_path,conversion_path)
    if canonical(schema_json)!=canonical(payload["schema_file"]):
        raise ValueError("Frozen schema content changed.")
    # Validate every source anchor before making the first policy prediction.
    inputs=[]
    with readonly_hdf5(dataset) as source:
        current=[decode(v) for v in source["mask/valid"][:5]]
        if current!=payload["selected_validation_episodes"]:
            raise ValueError("First-five validation selection changed.")
        for anchor in payload["anchors"]:
            observation,actual=extract_anchor(source["data"][anchor["source_episode"]],anchor["frame_index_zero_based"],schema,conversion,state_key,cameras)
            for key,value in actual.items():
                if canonical(value)!=canonical(anchor[key]):
                    raise ValueError(f"Frozen anchor source changed: {anchor['anchor_id']}/{key}")
            observation["_episode_policy_seed"]=anchor["noise_seed"]
            inputs.append(observation)
    if args.output_dir.exists():
        raise FileExistsError("Use a new model diagnostic directory; never overwrite previous predictions.")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/"frozen_anchors.json",envelope,exclusive=True)
    report={"created_at_utc":now(),"status":"started","model_label":args.model_label,
            "diagnostic_code_sha256":file_hash(__file__),
            "expected_model":args.expected_model,"expected_checkpoint":str(args.expected_checkpoint),
            "anchor_manifest_sha256":envelope["anchor_manifest_sha256"],"expected_anchors":15,
            "completed_anchors":0,"model_action_horizon":10,"real_action_dim":14,
            "idle_guard":idle,"interpretation":"Expert-observation held-out open-loop action imitation; not closed-loop success, unique correct action, convergence, or a blind final test.",
            "units":"Raw controller input numbers; not realized metres/radians. Clipped comparison clips both prediction and GT to audited bounds.",
            "metric_indices":{"pose":POSE,"translation":TRANSLATION,"rotation":ROTATION,"gripper":GRIPPER}}
    records=[]
    client=None
    try:
        client=LocalWireClient(args.port,args.inference_timeout)
        report["server_metadata"]=client.metadata
        report["metadata_validation"]=validate_metadata(client.metadata,schema,payload,args.expected_model,args.expected_checkpoint)
        write_json(args.output_dir/"server_metadata.json",client.metadata,exclusive=True)
        low=np.asarray(payload["action_bounds"]["low"],dtype=np.float32)
        high=np.asarray(payload["action_bounds"]["high"],dtype=np.float32)
        for anchor,observation in zip(payload["anchors"],inputs):
            check_idle()
            started=time.perf_counter()
            prediction=client.infer(observation)
            elapsed=(time.perf_counter()-started)*1000
            actions=np.asarray(prediction["actions"],dtype=np.float32)
            gt=np.asarray(anchor["gt_actions_raw_float32"],dtype=np.float32)
            metrics=action_metrics(actions,gt,low,high)
            record={"anchor_id":anchor["anchor_id"],"source_episode":anchor["source_episode"],
                    "frame_index_zero_based":anchor["frame_index_zero_based"],"noise_seed":anchor["noise_seed"],
                    "anchor_manifest_sha256":envelope["anchor_manifest_sha256"],
                    "prediction_raw14":actions.tolist(),"prediction_clipped14":np.clip(actions,low,high).tolist(),
                    "gt_raw14":gt.tolist(),"gt_clipped14":np.clip(gt,low,high).tolist(),
                    "metrics":metrics,"roundtrip_ms":elapsed,"policy_timing":prediction.get("policy_timing"),
                    "observation_contains_future_images_or_states":False}
            write_json(args.output_dir/(anchor["anchor_id"]+".json"),record,exclusive=True)
            records.append(record)
            report["completed_anchors"]=len(records)
        raw=np.asarray([r["prediction_raw14"] for r in records],dtype=np.float32)
        gt=np.asarray([r["gt_raw14"] for r in records],dtype=np.float32)
        report.update(status="complete",metrics=action_metrics(raw,gt,low,high),
                      per_episode_metrics={episode:action_metrics(
                          np.asarray([r["prediction_raw14"] for r in records if r["source_episode"]==episode],dtype=np.float32),
                          np.asarray([r["gt_raw14"] for r in records if r["source_episode"]==episode],dtype=np.float32),low,high)
                          for episode in payload["selected_validation_episodes"]})
    except Exception:
        report.update(status="failed",error=traceback.format_exc())
        raise
    finally:
        report["finished_at_utc"]=now()
        write_json(args.output_dir/"summary.json",report)
        if client is not None:
            client.close()
    return {"status":report["status"],"completed_anchors":len(records),"output":str(args.output_dir.resolve())}


def self_check(args):
    low=np.full(14,-1,dtype=np.float32);high=-low
    gt=np.zeros((10,14),dtype=np.float32)
    gt[:,6]=1;gt[:,13]=-1
    pred=gt.copy();pred[:,6]=2  # Gripper magnitude must not enter pose MAE.
    metric=action_metrics(pred,gt,low,high)
    assert metric["raw"]["pose12_mae"]==0 and metric["gripper"]["pooled"]["sign_accuracy"]==1
    pred[:,12]=.5  # Index 12 is robot1 rotation; [:12] would silently omit it.
    metric=action_metrics(pred,gt,low,high)
    assert np.isclose(metric["raw"]["pose12_mae"],.5/12)
    assert np.isclose(metric["raw"]["rotation6_mae"],.5/6) and metric["raw"]["translation6_mae"]==0
    gt[:,6]=np.resize(np.array([-1,0,1],dtype=np.float32),10)
    pred=gt.copy();metric=action_metrics(pred,gt,low,high)
    assert metric["gripper"]["pooled"]["sign_accuracy"]==1
    assert metric["gripper"]["robot0"]["gt_class_counts"]=={"open":4,"hold":3,"close":3}
    assert metric["gripper"]["robot0"]["always_open_accuracy"]==.4
    assert metric["gripper"]["robot0"]["always_close_accuracy"]==.3
    batched=action_metrics(np.stack([pred]*15),np.stack([gt]*15),low,high)
    assert batched["gripper"]["pooled"]["commands"]==300
    assert batched["gripper"]["robot0"]["commands"]==150
    transitions=gt.copy();transitions[:,6]=[-1]*5+[1]*5
    transition_metric=action_metrics(np.stack([transitions]*15),np.stack([transitions]*15),low,high)
    assert transition_metric["gripper"]["within_window_gt_sign_transitions_by_arm"]["robot0"]==15
    bad=pred.copy();bad[0,0]=np.nan
    try: action_metrics(bad,gt,low,high)
    except ValueError: pass
    else: raise AssertionError("Nonfinite actions accepted.")
    checks=["14D shape/finiteness","gripper excluded from pose indices","robot1 rotation index12 included",
            "translation/rotation split","strict open/hold/close signs","class counts and constant baselines",
            "15×10×2=300 gripper-command denominator","no artificial transitions between disjoint anchor windows"]
    fake_payload={"frames":list(FRAMES),"action_horizon":10,"anchors":[{} for _ in range(15)]}
    envelope={"anchor_manifest":fake_payload,"anchor_manifest_sha256":sha(canonical(fake_payload))}
    frozen_payload(envelope)
    changed=json.loads(json.dumps(envelope));changed["anchor_manifest"]["frames"][1]=101
    try: frozen_payload(changed)
    except ValueError: pass
    else: raise AssertionError("Modified frozen manifest accepted.")
    checks.append("frozen-manifest tampering rejected")
    if args.schema:
        import tempfile
        schema=TransportSchema.load(args.schema)
        if schema.state_dim!=18:
            raise ValueError("CPU contract check requires the real 18-D schema.")
        conversion={"state_fields":[{"hdf5_key":f"obs/robot{arm}_{name}","dimension":size}
                                    for arm in (0,1) for name,size in (("eef_pos",3),("eef_quat",4),("gripper_qpos",2))],
                    "task":"Synthetic CPU contract probe; never a model prediction.",
                    "image_shapes_hwc":{f"synthetic_camera_{i}":[2,2,3] for i in range(3)}}
        group={"actions":np.zeros((210,14),dtype=np.float32)}
        for field in conversion["state_fields"]:
            group[field["hdf5_key"]]=np.zeros((210,field["dimension"]),dtype=np.float32)
        cameras={slot:f"synthetic_camera_{i}" for i,slot in enumerate(CAMERA_SLOTS)}
        pixels=np.arange(12,dtype=np.uint8).reshape(2,2,3)
        for camera in cameras.values():
            group[f"obs/{camera}_image"]=np.tile(pixels,(210,1,1,1))
        state_key=read_json(args.schema).get("runtime",{}).get("combined_state_key","observation.state")
        observation,anchor=extract_anchor(group,100,schema,conversion,state_key,cameras)
        assert set(observation)=={state_key,schema.prompt_key,*schema.cameras.values()}
        assert observation[state_key].shape==(18,) and np.asarray(anchor["gt_actions_raw_float32"]).shape==(10,14)
        for key in schema.cameras.values():
            assert np.array_equal(observation[key],pixels) and not np.array_equal(observation[key],pixels[::-1])
        before=canonical({key:array_hash(value) if isinstance(value,np.ndarray) else value for key,value in observation.items()})
        for camera in cameras.values(): group[f"obs/{camera}_image"][101]=255
        observation_again,_=extract_anchor(group,100,schema,conversion,state_key,cameras)
        after=canonical({key:array_hash(value) if isinstance(value,np.ndarray) else value for key,value in observation_again.items()})
        assert before==after  # Altering a future image cannot alter the current policy payload.
        with tempfile.TemporaryDirectory(prefix="transport-actiondiag-cpu-") as temporary:
            norm=Path(temporary)/"norm_stats.json";norm.write_text('{"synthetic_cpu_probe":true}')
            payload={"norm_stats_path":str(norm),"norm_stats_sha256":file_hash(norm)}
            checkpoint=Path(temporary)/"base_checkpoint"
            metadata={**schema.summary(),"model":"base","checkpoint":str(checkpoint),"episode_seed_reset":True}
            validate_metadata(metadata,schema,payload,"base",checkpoint)
            wrong={**metadata,"state_dim":8}
            try: validate_metadata(wrong,schema,payload,"base",checkpoint)
            except ValueError: pass
            else: raise AssertionError("Wrong server schema accepted.")
            try: validate_metadata(metadata,schema,payload,"base",checkpoint/"wrong")
            except ValueError: pass
            else: raise AssertionError("Wrong server checkpoint accepted.")
        checks.extend(["real-schema current-only 3RGB/18state construction","unflipped RGB contract",
                       "future RGB changes do not enter current payload","wrong server schema/checkpoint rejected"])
    report={"status":"PASS","synthetic_only":True,"model_predictions_run":0,
            "checks":checks,
            "live_server_contract_exercised":False,"real_anchor_freeze_exercised":False}
    if args.report: write_json(args.report,report,exclusive=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("freeze",help="CPU-only immutable anchor selection before diagnostic predictions")
    for flag in ("dataset","schema","conversion-manifest","norm-stats","output"):
        p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--repo-id",default="local/transport_train")
    p.add_argument("--seed",type=int,default=101)
    p.set_defaults(handler=freeze)
    p=commands.add_parser("evaluate",help="Client for one already-running idle model server; never starts one")
    p.add_argument("--anchors",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--model-label",required=True)
    p.add_argument("--expected-model",choices=("base","lora"),required=True)
    p.add_argument("--expected-checkpoint",type=Path,required=True)
    p.add_argument("--port",type=int,default=8001)
    p.add_argument("--inference-timeout",type=float,default=180)
    p.set_defaults(handler=evaluate)
    p=commands.add_parser("self-check",help="Synthetic CPU metric checks, no network/model/GPU")
    p.add_argument("--report",type=Path)
    p.add_argument("--schema",type=Path,help="Also exercise current-only input and metadata with the real schema, on synthetic arrays")
    p.set_defaults(handler=self_check)
    args=parser.parse_args()
    print(json.dumps(args.handler(args),indent=2,ensure_ascii=False,default=json_default))


if __name__=="__main__":
    main()
