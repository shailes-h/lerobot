# GR00T N1.7 modality config for BimanualYAM (joint space), modeled on examples/SO100/so100_config.py.
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ARM = ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
GRIPPER = ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
PARTS = ["left_arm", "left_gripper", "right_arm", "right_gripper"]

yam_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["top", "left", "right"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=PARTS),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=PARTS,
        action_configs=[ARM, GRIPPER, ARM, GRIPPER],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(yam_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
