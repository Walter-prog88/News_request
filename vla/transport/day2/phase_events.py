"""Pure numeric contact-event rules for train-only expert phase annotations."""
from __future__ import annotations
import numpy as np

METHOD = 'physical_action_replay_finger_contacts_v2'


def first_sustained(values, start=0, hold=3):
    values = np.asarray(values, dtype=bool)
    if values.ndim != 1 or hold < 1:
        raise ValueError('Require a one-dimensional event signal and positive hold')
    for step in range(max(0, start), len(values) - hold + 1):
        if values[step:step + hold].all():
            return step
    return None


def handover_events(contact, payload_z, payload_in_target, frame_count, success_ever,
                    donor=0, receiver=1, hold_steps=3, margin_before=20, margin_after=20):
    """Input rows are physical states after 0..T original control actions.

    Contact is each Panda fingerpad group's geometric contact, not force closure.
    A valid transition requires donor contact, then sustained receiver contact
    with the payload lifted, followed by donor contact ending while receiver
    contact remains. No gripper commands are inspected.
    """
    raw_contact = np.asarray(contact)
    if not np.isin(raw_contact, [0, 1]).all():
        raise ValueError('Contact predicates must be binary and finite')
    contact = raw_contact.astype(bool)
    payload_z = np.asarray(payload_z, dtype=np.float64)
    target = np.asarray(payload_in_target)
    if not np.isin(target, [0, 1]).all():
        raise ValueError('Target contact predicates must be binary and finite')
    if contact.shape != (frame_count + 1, 2, 2) or payload_z.shape != (frame_count + 1,) or target.shape != (frame_count + 1,):
        raise ValueError('Expected initial state plus one row after every action')
    if not np.isfinite(payload_z).all() or donor == receiver or {donor, receiver} != {0, 1}:
        raise ValueError('Invalid payload trajectory or arm identities')
    bilateral = contact.all(axis=-1)
    # Use the same five-centimetre lift reference as conditional evaluation.
    # Never label a contact transition after the payload has reached the target.
    eligible = ~np.maximum.accumulate(target.astype(bool))
    lifted = payload_z >= payload_z[0] + 0.05
    donor_first = first_sustained(bilateral[:, donor] & eligible, hold=hold_steps)
    receiver_first = None
    receiver_signal = bilateral[:, receiver] & lifted & eligible
    if donor_first is not None:
        for candidate in range(donor_first + hold_steps - 1, frame_count - hold_steps + 2):
            # A direct handover needs simultaneous bilateral contact. The
            # supplying hand must still be in its own sustained contact segment.
            # Historical donor contact followed by a table regrasp is excluded.
            donor_segment = bilateral[candidate - hold_steps + 1:candidate + 1, donor]
            if receiver_signal[candidate:candidate + hold_steps].all() and donor_segment.all():
                receiver_first = candidate
                break
    release = None
    if receiver_first is not None:
        breaks = np.flatnonzero(~receiver_signal[receiver_first:])
        segment_end = receiver_first + int(breaks[0]) if len(breaks) else len(receiver_signal)
        release = first_sustained(
            ~contact[:segment_end, donor].any(axis=-1) & receiver_signal[:segment_end],
            start=receiver_first, hold=hold_steps)
    valid = bool(success_ever and receiver_first is not None and release is not None)
    interval = None
    if valid:
        # A state after control action k follows dataset action frame k-1.
        interval = [max(0, receiver_first - 1 - margin_before),
                    min(frame_count, release - 1 + margin_after + 1)]
        valid = interval[0] < interval[1]
    return {'valid_handover_window': valid, 'donor_robot_index': donor, 'receiver_robot_index': receiver,
            'first_sustained_donor_contact_step': donor_first,
            'first_sustained_receiver_contact_lifted_step': receiver_first,
            'first_sustained_donor_release_receiver_contact_step': release,
            'donor_contact_confirmed_step': None if donor_first is None else donor_first + hold_steps - 1,
            'receiver_contact_lifted_confirmed_step': None if receiver_first is None else receiver_first + hold_steps - 1,
            'donor_all_fingers_clear_confirmed_step': None if release is None else release + hold_steps - 1,
            'simultaneous_bilateral_contact_step': receiver_first,
            'event_step_semantics': 'first_sustained fields are segment onset; confirmed fields are the last of the required consecutive samples. Three samples at20Hz span0.1s.',
            'lift_reference': 'Payload root-body world z at initial state plus0.05m; events must precede any payload-target-bin contact.',
            'focus_frame_interval_exclusive_end': interval,
            'hold_control_steps': hold_steps, 'margin_before_frames': margin_before,
            'margin_after_frames': margin_after,
            'reason_if_invalid': None if valid else (
                'Expert task did not succeed' if not success_ever else
                'No sustained supplying-hand bilateral contact before target contact' if donor_first is None else
                'No direct bilateral-overlap transition while lifted before target contact; table regrasp excluded' if receiver_first is None else
                'No complete donor-finger release in the same uninterrupted receiver contact/lift segment before target contact'),
            'event_boundary': 'Both-finger collision contact proxy and payload height; not measured grasp force or a proof of stable force closure.'}
