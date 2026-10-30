"""Train-demo grasp windows from original commands plus independent physical replay."""
from __future__ import annotations
import numpy as np

METHOD = 'original_close_transition_with_replayed_grasp_lift_v1'


def grasp_event(actions, contact, payload_z, target, *, success_ever,
                hold=3, max_close_to_contact=40, margin_before=10, margin_after=10,
                action_horizon=10):
    """Action frame t maps physical state t to t+1; no off-policy relabeling.

    Find the first donor bilateral segment that reaches sustained +5cm lift
    before payload-target contact. Require a preceding open→close command,
    allowing intervening zero/hold commands but no subsequent reopen before
    lift confirmation. The close must occur <=40 frames before contact onset.
    """
    actions = np.asarray(actions, dtype=np.float32)
    raw_contact = np.asarray(contact)
    z = np.asarray(payload_z, dtype=float)
    target = np.asarray(target)
    n = len(actions)
    if actions.shape != (n,14) or raw_contact.shape != (n+1,2,2) or z.shape != (n+1,) or target.shape != (n+1,):
        raise ValueError('Require original T×14 actions and T+1 physical state records')
    if not np.isfinite(actions).all() or not np.isfinite(z).all() or not np.isin(raw_contact,[0,1]).all() or not np.isin(target,[0,1]).all():
        raise ValueError('Nonfinite commands/pose or nonbinary events')
    if hold < 1 or action_horizon < 2 or min(max_close_to_contact,margin_before,margin_after) < 0:
        raise ValueError('Invalid event/window configuration')
    sign = np.sign(actions[:,6]).astype(np.int8)
    closes = []
    last_nonzero = 0
    for t, value in enumerate(sign):
        if value == 1 and last_nonzero == -1:
            closes.append(t)
        if value != 0:
            last_nonzero = int(value)
    eligible = ~np.maximum.accumulate(target.astype(bool))
    signal = raw_contact.astype(bool)[:,0].all(axis=-1) & eligible
    # Every maximal donor bilateral segment is considered once, in time order.
    starts = np.flatnonzero(signal & ~np.r_[False, signal[:-1]])
    result = {'method': METHOD, 'valid_grasp_window': False,
              'reason_if_invalid': 'Expert replay did not succeed' if not success_ever else 'No close→continuous bilateral-contact→sustained lift sequence',
              'all_donor_open_to_close_action_frames': closes,
              'focus_frame_interval_exclusive_end': None,
              'transition_target_anchor_frames': [],
              'event_boundary': 'Contact/height proxy, not grasp force closure. Original observations and actions remain paired; event timing is inferred from independent action replay.'}
    if not success_ever:
        return result
    for onset in starts:
        breaks = np.flatnonzero(~signal[onset:])
        end = int(onset + breaks[0]) if len(breaks) else n+1
        lift = None
        for k in range(int(onset), end-hold+1):
            if (z[k:k+hold] >= z[0]+0.05).all():
                lift = k
                break
        if lift is None or end-onset < hold:
            continue
        candidates = [c for c in closes if 0 <= int(onset)-1-c <= max_close_to_contact
                      and not (sign[c:min(n,lift+hold-1)] < 0).any()]
        if not candidates:
            continue
        close = candidates[-1]
        confirm = lift+hold-1
        start, stop = max(0,close-margin_before), min(n,confirm+margin_after)
        # Require each anchor's H10 target to contain both actual negative/open
        # and positive/close commands around this specific close transition.
        transition_anchors = []
        for anchor in range(max(start,close-action_horizon+1), min(close+1,stop)):
            chunk = sign[anchor:min(n,anchor+action_horizon)]
            if (chunk < 0).any() and (chunk > 0).any():
                transition_anchors.append(anchor)
        if not transition_anchors:
            continue
        return {**result, 'valid_grasp_window': True, 'reason_if_invalid': None,
                'close_action_frame': close, 'donor_bilateral_onset_state': int(onset),
                'donor_bilateral_segment_end_state_exclusive': end,
                'lift_onset_state': lift, 'lift_confirmed_state': confirm,
                'focus_frame_interval_exclusive_end': [start,stop],
                'transition_target_anchor_frames': transition_anchors,
                'transition_anchor_definition': 'Original H10 donor gripper targets contain negative/open and positive/close commands around selected close; zero means hold.'}
    return result
