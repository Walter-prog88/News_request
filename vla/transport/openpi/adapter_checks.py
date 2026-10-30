"""Small CPU checks for data corruption and wrong action-space regressions."""
import dataclasses
import json
from pathlib import Path
import tempfile

import numpy as np

from transport_policy import TransportInputs, TransportOutputs, TransportSchema, StateField


def main():
    schema = TransportSchema(
        cameras={"base_0_rgb": "cam_external", "left_wrist_0_rgb": "cam_arm_a",
                 "right_wrist_0_rgb": "cam_arm_b"},
        state_fields=(StateField("q", 3, "arm_a", "synthetic joint positions", (0, 2, 4)),
                      StateField("q", 3, "arm_b", "synthetic joint positions", (1, 3, 5))),
        action_key="action", arm_order=("arm_a", "arm_b"),
        action_description="Synthetic adapter check, not demonstration evidence", fps=20,
        audit_source="synthetic CPU check only",
    )
    schema.validate()
    action = np.linspace(-.9, .8, 42, dtype=np.float32).reshape(3, 14)
    sample = {"q": np.arange(6, dtype=np.float32), "action": action.copy(),
              "cam_external": np.full((12, 16, 3), 127, dtype=np.uint8),
              "cam_arm_a": np.zeros((3, 12, 16), dtype=np.float32),
              "cam_arm_b": np.ones((3, 12, 16), dtype=np.float32),
              "prompt": "synthetic transport check", "privileged_object_pose": np.ones(7)}
    result = TransportInputs(schema)(sample)
    np.testing.assert_array_equal(result["actions"], action)
    np.testing.assert_array_equal(sample["action"], action)
    np.testing.assert_array_equal(result["state"], [0, 2, 4, 1, 3, 5])
    assert "privileged_object_pose" not in result
    assert result["image"]["left_wrist_0_rgb"].max() == 0
    assert result["image"]["right_wrist_0_rgb"].min() == 255
    padded = np.pad(action, ((0, 0), (0, 18)))
    np.testing.assert_array_equal(TransportOutputs()({"actions": padded})["actions"], action)
    missing_camera = {key: value for key, value in sample.items() if key != "cam_arm_b"}
    try:
        TransportInputs(schema)(missing_camera)
    except KeyError:
        pass
    else:
        raise AssertionError("Missing wrist camera was silently padded.")
    malformed = {**sample, "action": np.zeros((3, 16), dtype=np.float32)}
    try:
        TransportInputs(schema)(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong action width was accepted.")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "schema.json"
        content = {**dataclasses.asdict(schema), "schema_version": 1, "audited": False,
                   "action_dim": 14, "action_representation": "relative_eef_controller_command"}
        path.write_text(json.dumps(content))
        try:
            TransportSchema.load(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Unaudited schema was accepted.")
        content["audited"] = True
        path.write_text(json.dumps(content))
        loaded = TransportSchema.load(path)
        np.testing.assert_array_equal(TransportInputs(loaded)(sample)["state"], result["state"])
    print("PASS: action scale/order preserved; two-arm state order explicit; three RGB views required; privileged fields excluded; unaudited schema blocked.")


if __name__ == "__main__":
    main()
