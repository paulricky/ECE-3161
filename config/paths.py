from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
ROBOT_DIR = PROJECT_ROOT / "robot"
VISION_DIR = PROJECT_ROOT / "vision"
PICK_PLACE_DIR = PROJECT_ROOT / "pick_place"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

CALIBRATION_DATA_DIR = PROJECT_ROOT / "calibration_data"
CALIBRATION_ARTIFACTS_DIR = PROJECT_ROOT / "calibration_artifacts"

ROBOT_JOINT_CALIBRATION_FILE = CALIBRATION_DATA_DIR / "robot_joint_calibration.json"
ROBOT_MOTOR_SETUP_FILE = CALIBRATION_DATA_DIR / "robot_motor_setup.json"

ROBOT_MODEL_CALIBRATION_FILE = CALIBRATION_ARTIFACTS_DIR / "robot_model_calibration.json"
KINEMATIC_MODEL_FILE = CALIBRATION_ARTIFACTS_DIR / "kinematic_model.json"

URDF_DIR = CALIBRATION_ARTIFACTS_DIR / "urdf"
MESH_DIR = CALIBRATION_ARTIFACTS_DIR / "meshes"
