CAM_W, CAM_H = 720, 720

# Smoothing (0..1)
JOINT_SMOOTH_ALPHA = 0.20
ANGLE_SMOOTH_ALPHA = 0.25

# Physics
# SIM_HZ = 240
# MOTOR_FORCE = 800
# POS_GAIN = 0.35
# VEL_GAIN = 1.00

# Log overlay
LOG_MAX = 12
LOG_DURATION = 6.0

# Hand openness and ArUco glove
OPEN_FINGER_EXTENDED_ON = 0.62
OPEN_FINGER_EXTENDED_OFF = 0.52
OPEN_HAND_MIN_OPEN_FINGERS = 3
OPEN_HAND_MAX_CLOSED_FINGERS = 1
OPEN_HAND_PINCH_BLOCK = 0.10
HAND_STATE_SMOOTHING = 0.20

ARUCO_GLOVE_ENABLED = True
ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_GLOVE_FRONT_ID = 1
ARUCO_GLOVE_BACK_ID = 5
ARUCO_MARKER_SIZE_M = 0.03

# Hand tracking sensitivity (direct-joint-mapping bounds).
# Each joint is driven from one screen axis: hand_cx -> shoulder_pan,
# hand_cy -> shoulder_lift, palm_size -> elbow_flex. These software bounds
# are intentionally wider than the calibrated motor range so the servo's
# physical limit is what saturates first. A full sweep of the central portion
# of the camera frame produces a full arm sweep.
BASE_PAN_MIN = -3.0
BASE_PAN_MAX = 3.0
SHOULDER_LIFT_MIN = -2.0
SHOULDER_LIFT_MAX = 2.0
ELBOW_MIN = -0.5
ELBOW_MAX = 2.5
WRIST_FLEX_MIN = -2.5
WRIST_FLEX_MAX = 2.5
WRIST_YAW_MIN = -3.141592653589793
WRIST_YAW_MAX = 3.141592653589793
WRIST_ROLL_MIN = -3.141592653589793
WRIST_ROLL_MAX = 3.141592653589793
WRIST_PITCH_MIN = -2.5
WRIST_PITCH_MAX = 2.5

# Clap
CLAP_COOLDOWN_S = 0.6
CLAP_CLOSE_ENOUGH = 0.12
CLAP_FAST_CLOSING = 0.35

# Snap
SNAP_COOLDOWN_S = 0.5
SNAP_PINCH_ON = 0.045
SNAP_PINCH_OFF = 0.075
SNAP_FAST_RELEASE = 0.30

# Gripper mapping (thumb-index pinch)
PINCH_MIN = 0.01
PINCH_MAX = 0.95

# Reach mapping (wrist to middle MCP)
REACH_MIN = 0.05
REACH_MAX = 0.95

# Inversions
INVERT_SHOULDER_LIFT = False
INVERT_BASE_PAN = False
INVERT_ELBOW = True
INVERT_WRIST_FLEX = False
INVERT_WRIST_YAW = False
INVERT_WRIST_ROLL = False
INVERT_WRIST_PITCH = False
INVERT_GRIPPER = False

# Hand-tracking workspace bounds for the 2-DOF analytic IK in the vertical
# plane. Corners must fit inside the arm's ~0.28 m reach from the shoulder
# (located at z = IK_SHOULDER_Z_M). X is unused because shoulder_pan is now
# direct-mapped from hand_cx.
WORKSPACE_X_MIN = -0.12
WORKSPACE_X_MAX = 0.12

# y: forward distance (larger palm in frame -> arm pulls back, smaller y)
WORKSPACE_Y_MIN = 0.10
WORKSPACE_Y_MAX = 0.22

# z: end-effector world height (top of frame -> larger z)
WORKSPACE_Z_MIN = 0.00
WORKSPACE_Z_MAX = 0.22

# IK controls
IK_MAX_ITERS = 60
IK_RESIDUAL_THRESH = 1e-4
IK_DLS_DAMPING = 0.08
IK_DLS_POSITION_GAIN = 1.0
IK_DLS_PITCH_GAIN = 0.35
IK_DLS_ORIENTATION_GAIN = 0.45
IK_DLS_CONTINUITY_GAIN = 0.08
IK_DLS_MAX_ITERS = 12
IK_DLS_MAX_STEP_RAD = 0.20

# Tool-link split for the 7-DOF chain: wrist_roll --[IK_TOOL_A_M]--> wrist_pitch --[IK_TOOL_B_M]--> EE.
# When wrist_pitch = 0, the two links lie collinear and act as the old single
# tool link of length (IK_TOOL_A_M + IK_TOOL_B_M). Defaults to a 25/25 mm split;
# measure and retune once the physical link lengths are known.
IK_TOOL_A_M = 0.025
IK_TOOL_B_M = 0.025
# Backward-compatible kinematic aliases used by legacy fallback paths.
IK_BASE_HEIGHT_M = 0.06
IK_SHOULDER_Z_M = IK_BASE_HEIGHT_M
IK_LINK1_M = 0.115
IK_LINK2_M = 0.115
IK_WRIST_TO_YAW_M = 0.0
IK_TOOL_M = IK_TOOL_A_M + IK_TOOL_B_M
IK_ELBOW_PREFERRED_WORLD_M = None

# Smooth target pose a bit (0..1)
POSE_SMOOTH_ALPHA = 0.75

CLOSED_TIPS_ON = 0.060
CLOSED_TIPS_OFF = 0.090

# Replace with your own file path
URDF_PATH = "/Users/ricky/PycharmProjects/ECE3161/SO-ARM100"

# Real robot control. Motion, torque, and speed-percent limits are defined in
# the final runtime sections below so there is only one effective override path.
ENABLE_REAL_ROBOT = True
REAL_ROBOT_PORT = ""
REAL_ROBOT_ID = "my_awesome_follower_arm"

# Use the custom calibration script for the 8-motor arm rather than the stock SO101 auto-calibration flow.
# The upstream SO follower driver defaults to a 6-motor SO100/SO101 bus, so
# robot_controller.py replaces that bus with the configured motor list below.
REAL_ROBOT_AUTO_CALIBRATE = False

# Use this project's direct Feetech controller for the custom 8-motor arm.
# This bypasses LeRobot's stock calibration loader, which expects LeRobot-native
# calibration objects and can fail on this project's richer JSON calibration file.
REAL_ROBOT_USE_DIRECT_PROJECT_CONTROLLER = True

# Full motor order for the modified arm: 7 controllable arm DOF + gripper.
# Motor IDs must match the physical servo IDs on the Feetech daisy chain.
REAL_ROBOT_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "wrist_pitch",
    "gripper",
]
REAL_ROBOT_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
REAL_ROBOT_MOTOR_MODEL = "sts3215"
REAL_ROBOT_MOTOR_MODELS = ["sts3215"] * len(REAL_ROBOT_MOTOR_NAMES)
REAL_ROBOT_MOTOR_MODEL_NUMBER = 777
REAL_ROBOT_MOTOR_MODEL_NUMBERS = [REAL_ROBOT_MOTOR_MODEL_NUMBER] * len(REAL_ROBOT_MOTOR_NAMES)
REAL_ROBOT_BAUDRATE = 1000000

# Fine-tuning offsets applied in software after the main calibration is loaded.
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, wrist_pitch
# Do not include the gripper here because it is mapped separately as 0..100%.
REAL_ROBOT_JOINT_OFFSETS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Project-local calibration file produced by robot_calibrate.py
ROBOT_JOINT_CALIBRATION_FILE = "calibration_data/robot_joint_calibration.json"
ROBOT_MODEL_CALIBRATION_FILE = "calibration_data/robot_model_calibration.json"

# Optional: if you want the custom calibration script to also mirror the same JSON into
# a LeRobot cache path for an installed follower driver, set this to that full path.
# Leave empty to skip the extra copy.
LEROBOT_DRIVER_CALIBRATION_FILE = ""

CALIB_INTRINSICS_FILE = "calibration_data/calibration_intrinsics.npz"
CALIB_EXTRINSICS_FILE = "calibration_data/calibration_extrinsics.npz"
CALIB_WORKSPACE_FILE = "calibration_data/calibration_workspace.npz"
CAMERA_CALIBRATION_FILE = "calibration_data/camera_calibration.npz"
CAMERA_CALIBRATION_JSON = "calibration_data/camera_calibration.json"
CAMERA_INTRINSICS_OPTIONAL = True
CAMERA_UNDISTORT_RUNTIME = False
CHARUCO_ENABLED = True
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH_M = 0.030
CHARUCO_MARKER_LENGTH_M = 0.022
CHARUCO_DICTIONARY = "DICT_4X4_50"
CHARUCO_MIN_MARKERS = 4
CHARUCO_MIN_CORNERS = 8
CHARUCO_REQUIRED_FRAMES = 25
CHARUCO_CAPTURE_DELAY_S = 0.4
CHARUCO_REPROJECTION_ERROR_WARN = 1.0
CHARUCO_CAMERA_INDEX = 0
CHARUCO_FRAME_WIDTH = 1280
CHARUCO_FRAME_HEIGHT = 720
CHARUCO_FPS = 30

EXTRINSICS_MODE = "workspace_to_camera"

ARUCO_WORKSPACE_MIN = (-0.18, -0.12, 0.02)
ARUCO_WORKSPACE_MAX = (0.18, 0.18, 0.28)

HAND_CMD_SMOOTHING = 0.6

# Math/hand-tracking calibration source. By default this points at the project-local robot calibration generated by robot_calibrate.py.
LEROBOT_CALIBRATION_FILE = ROBOT_JOINT_CALIBRATION_FILE


# ---- Pick-and-place: object detection ----
PICKPLACE_YOLO_MODEL_PATH = "yolov8n.pt"
PICKPLACE_YOLO_DEVICE = "mps"
PICKPLACE_YOLO_INFERENCE_HZ = 8.0
PICKPLACE_YOLO_CONF_THRESHOLD = 0.45
PICKPLACE_YOLO_CLASS_WHITELIST = [
    "cup", "bottle", "apple", "orange",
    "banana", "scissors", "cell phone", "mouse",
]

# ---- Pick-and-place: top-down camera ----
PICKPLACE_CAMERA_INDEX = 1
PICKPLACE_TOPDOWN_INTRINSICS_FILE = "calibration_data/topdown_intrinsics.npz"
PICKPLACE_TOPDOWN_EXTRINSICS_FILE = "calibration_data/topdown_extrinsics.npz"
PICKPLACE_CALIB_CHESSBOARD_ROWS = 7
PICKPLACE_CALIB_CHESSBOARD_COLS = 9
PICKPLACE_CALIB_SQUARE_SIZE_M = 0.025
PICKPLACE_CALIB_MIN_INTRINSICS_CAPTURES = 12
PICKPLACE_CALIB_ANCHOR_MARKER_SIZE_M = 0.04
PICKPLACE_CALIB_ANCHOR_MARKER_DICT = "DICT_6X6_250"
PICKPLACE_CALIB_ANCHOR_MAP = [
    [0, [-0.15, 0.10, 0.00]],
    [1, [ 0.15, 0.10, 0.00]],
    [2, [-0.15, 0.30, 0.00]],
    [3, [ 0.15, 0.30, 0.00]],
]
PICKPLACE_CALIB_STABLE_FRAMES = 20
PICKPLACE_CALIB_REPROJ_WARN_PX = 3.0
PICKPLACE_RECTANGLE_WORKSPACE_FILE = "calibration_data/topdown_rectangle_workspace.npz"

# ---- Pick-and-place: drop target marker ----
PICKPLACE_DROP_MARKER_ID = 42
PICKPLACE_DROP_MARKER_DICT = "DICT_4X4_50"
PICKPLACE_DROP_MARKER_SIZE_M = 0.04

# ---- Pick-and-place: geometry ----
PICKPLACE_TABLE_Z_M = 0.02
PICKPLACE_PICK_Z_OFFSET_M = 0.005
PICKPLACE_PLACE_Z_OFFSET_M = 0.015
PICKPLACE_TRANSIT_Z_OFFSET_M = 0.08

# ---- Pick-and-place: IK / grasp orientation ----
IK_TOPDOWN_APPROACH_OFFSET_RAD = 0.05
IK_ABORT_POSITION_ERR_M = 0.010
PICKPLACE_HOME_JOINTS_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PICKPLACE_HOME_GRIPPER_OPEN01 = 1.0

# ---- Pick-and-place: state machine ----
PICKPLACE_ARRIVAL_TOL_RAD = 0.03
PICKPLACE_WAYPOINT_DWELL_S = 0.3
PICKPLACE_GRIPPER_CLOSE_DWELL_S = 0.8
PICKPLACE_GRIPPER_OPEN_DWELL_S = 0.5
PICKPLACE_WAYPOINT_TIMEOUT_S = 8.0
PICKPLACE_MAX_ABORTS_BEFORE_ERROR = 3
PICKPLACE_CONTINUOUS_MODE = True
PICKPLACE_SCAN_DETECTION_REQUIRED_FRAMES = 2
PICKPLACE_PICKING_POLICY = "highest_confidence"

# ---- Direct Feetech calibration diagnostics/read behavior ----
FEETECH_MODEL_NUMBER_ADDR = 3
FEETECH_ID_REGISTER_ADDR = 5
FEETECH_TORQUE_ENABLE_ADDR = 40
FEETECH_GOAL_POSITION_ADDR = 42
FEETECH_PRESENT_POSITION_ADDR = 56
FEETECH_PRESENT_POSITION_LEN = 2

# Baudrates to try when FEETECH_CAPTURE_FULL_BAUD_SCAN is enabled.
REAL_ROBOT_SCAN_BAUDRATES = [
    1000000,
    500000,
    250000,
    128000,
    115200,
    57600,
    38400,
    19200,
]

# ---- No-freeze identify/calibration defaults ----
# Keep these defaults quick. Turn the full scans on only for intentional slow diagnostics.
FEETECH_STARTUP_CHAIN_VERIFY = False
FEETECH_CAPTURE_FULL_BAUD_SCAN = False
FEETECH_IDENTIFY_FULL_BAUD_SCAN = False
FEETECH_IDENTIFY_READ_RETRIES = 2
FEETECH_IDENTIFY_INTER_RETRY_S = 0.015
FEETECH_IDENTIFY_POSITION_ADDR_CANDIDATES = [56]
FEETECH_CAPTURE_READ_RETRIES = 2
FEETECH_CAPTURE_INTER_RETRY_S = 0.015
FEETECH_CAPTURE_POSITION_ADDR_CANDIDATES = [56]
FEETECH_PRESENT_POSITION_ADDR_CANDIDATES = [56]
FEETECH_DIRECT_PACKET_TIMEOUT_S = 0.18
FEETECH_LIVE_TABLE_MAX_SECONDS = 0.0


# ---- No-freeze motor setup defaults ----
# Keep setup scans short. Turn full scans on only when a motor has an unknown ID/baud.
FEETECH_SETUP_FULL_BAUD_SCAN = False
FEETECH_SETUP_FULL_ID_SCAN = False
FEETECH_SETUP_SCAN_MAX_ID = 20
FEETECH_SETUP_EXTRA_SCAN_IDS = [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
FEETECH_SETUP_PROBE_TIMEOUT_S = 0.07
FEETECH_SETUP_INTER_PROBE_DELAY_S = 0.005

# Stable persistent motor-bus communication. This keeps the serial port open
# during identify/calibration instead of reopening it for every motor read.
FEETECH_STABLE_BUS_ENABLED = True
FEETECH_STABLE_BUS_PACKET_TIMEOUT_S = 0.10
FEETECH_STABLE_BUS_READ_RETRIES = 5
FEETECH_STABLE_BUS_WRITE_RETRIES = 3
FEETECH_STABLE_BUS_INTER_PACKET_DELAY_S = 0.006
FEETECH_STABLE_BUS_DRAIN_BEFORE_TX = True
FEETECH_STABLE_IDENTIFY_ATTEMPTS = 3
FEETECH_STABLE_BAUD_SWITCH_DELAY_S = 0.08



# ---- Read-only ID scan defaults ----
# Option 5 uses these settings. It never writes/flashes IDs.
FEETECH_READONLY_SCAN_MAX_ID = 20
FEETECH_READONLY_SCAN_EXTRA_IDS = [0]
FEETECH_READONLY_SCAN_FULL_BAUD_SCAN = False
FEETECH_READONLY_SCAN_READ_RETRIES = 2
FEETECH_READONLY_SCAN_SHOW_MISSES = False
# ---- Safer Feetech/ST3215 ID setup and read-only scan behavior ----
# These final assignments intentionally override earlier diagnostic defaults.
# Setup no longer trusts write status packets alone; it rescans after every ID write
# and only saves metadata after exactly the target ID responds.
FEETECH_SETUP_CONFIRM_REASSIGN_VALID_ID = True
FEETECH_SETUP_VERIFY_EXACTLY_ONE_ID = True
FEETECH_SETUP_VERIFY_RESCAN_AFTER_WRITE = True
FEETECH_SETUP_POWER_CYCLE_AFTER_ID_WRITE = False
FEETECH_SETUP_READ_RETRIES = 2
FEETECH_SETUP_WRITE_VERIFY_RETRIES = 8
FEETECH_SETUP_POST_WRITE_DELAY_S = 0.8
FEETECH_DIRECT_WRITE_RETRIES = 3

# Keep normal diagnostics fast. Turn these on only when intentionally scanning unknown hardware.
FEETECH_SETUP_FULL_BAUD_SCAN = False
FEETECH_SETUP_FULL_ID_SCAN = False
FEETECH_SETUP_SCAN_MAX_ID = 20
FEETECH_READONLY_SCAN_MAX_ID = 20
FEETECH_READONLY_SCAN_EXTRA_IDS = [0]
FEETECH_READONLY_SCAN_FULL_BAUD_SCAN = False
FEETECH_READONLY_SCAN_READ_RETRIES = 2
FEETECH_READONLY_SCAN_SHOW_MISSES = False

# Stable persistent motor-bus reads for identify/calibration.
FEETECH_STABLE_IDENTIFY_ATTEMPTS = 3
FEETECH_IDENTIFY_FULL_BAUD_SCAN = False
FEETECH_DIRECT_PACKET_TIMEOUT_S = 0.18
FEETECH_DIRECT_READ_RETRIES = 3
FEETECH_STABLE_BUS_INTER_PACKET_DELAY_S = 0.006
FEETECH_LIVE_TABLE_PERIOD_S = 0.50
# ---------------------------------------------------------------------------
# Safe one-motor ID flashing / verification
# ---------------------------------------------------------------------------
# STS/SMS/Feetech servo ID is stored in EEPROM. Unlock before writing ID and
# verify by a read-only scan afterward. The script must not trust write ACKs.
FEETECH_LOCK_REGISTER_ADDR = 55
FEETECH_LOCK_UNLOCK_VALUE = 0
FEETECH_LOCK_LOCK_VALUE = 1
FEETECH_ID_FLASH_VERIFY_RETRIES = 10
FEETECH_ID_FLASH_VERIFY_DELAY_S = 0.30
FEETECH_ID_FLASH_REQUIRE_ID_REGISTER_MATCH = True
FEETECH_ID_FLASH_LOCK_EEPROM_AFTER_WRITE = True
FEETECH_SETUP_POST_WRITE_DELAY_S = 0.80
FEETECH_SETUP_WRITE_VERIFY_RETRIES = 10
# Direct Feetech controller communication timing.
DIRECT_FEETECH_INTER_WRITE_DELAY_S = 0.004
DIRECT_FEETECH_READ_TIMEOUT_S = 0.08
# ---- 7-DOF calibrated IK / null-space tuning ----
IK_MODEL_CALIBRATION_FILE = "calibration_data/robot_model_calibration.json"
IK_DLS_STEP_GAIN = 0.85
IK_DLS_ORIENTATION_TOL = 0.002
IK_REACHABLE_POSITION_ERR_M = 0.020
IK_REACHABLE_ORIENTATION_ERR_RAD = 0.35
IK_COST_POSITION = 1.0
IK_COST_ORIENTATION = 0.25
IK_COST_CONTINUITY = 0.05
IK_COST_LIMIT = 0.02
IK_COST_POSTURE = 0.01
IK_COST_COLLISION = 0.0
IK_NULLSPACE_LIMIT_GAIN = 0.025
IK_NULLSPACE_CONTINUITY_GAIN = 0.035
IK_NULLSPACE_POSTURE_GAIN = 0.020
IK_NULLSPACE_COLLISION_GAIN = 0.0
IK_PREFERRED_POSTURE_RAD = [0.0, 0.0, 0.65, 0.0, 0.0, 0.0, 0.0]
IK_ENABLE_SELF_COLLISION_COST = False
IK_SELF_COLLISION_SAFE_DIST_M = 0.035
HAND_IK_REJECT_POSITION_ERR_M = 0.060
HAND_TARGET_PITCH_BIAS_RAD = -0.15
PICKPLACE_VISUAL_SERVO_ENABLED = True
PICKPLACE_VISUAL_SERVO_GAIN = 0.65
PICKPLACE_VISUAL_SERVO_MAX_CORRECTION_M = 0.045
PICKPLACE_VISUAL_SERVO_GRASP_READY_M = 0.015
REAL_ROBOT_ENABLE_TRACKING_WATCHDOG = True
REAL_ROBOT_TRACKING_WARN_DEG = 30.0
REAL_ROBOT_TRACKING_ABORT_DEG = 60.0
RESIDUAL_LEARNING_ENABLED = False
RESIDUAL_MODEL_FILE = "calibration_data/residual_model.json"
RESIDUAL_MAX_TRANSLATION_M = 0.020
RESIDUAL_MAX_ROTATION_RAD = 0.12
HAND_DEPTH_AUTOCALIBRATE = False
HAND_DEPTH_AUTOCALIBRATE_WINDOW = 240
HAND_DEPTH_AUTOCALIBRATE_EMA_ALPHA = 0.15
HAND_USE_CARTESIAN_IK = True
HAND_CARTESIAN_MAPPING_ENABLED = True
HAND_CARTESIAN_MAPPING_FALLBACK_TO_OLD = True
HAND_USE_WRIST_ORIENTATION = True
HAND_WRIST_ORIENTATION_FALLBACK_TO_NEUTRAL = True
HAND_CAMERA_AXIS_DEBUG = True
HAND_TARGET_X_MIN_M = WORKSPACE_X_MIN
HAND_TARGET_X_MAX_M = WORKSPACE_X_MAX
HAND_TARGET_Y_MIN_M = WORKSPACE_Y_MIN
HAND_TARGET_Y_MAX_M = WORKSPACE_Y_MAX
HAND_TARGET_Z_MIN_M = WORKSPACE_Z_MIN
HAND_TARGET_Z_MAX_M = WORKSPACE_Z_MAX
HAND_IMAGE_X_FLIP = False
HAND_IMAGE_Y_FLIP = True
HAND_DEPTH_FLIP = False
HAND_CAMERA_TO_ROBOT_AXIS_MAP = {"image_x": "robot_x", "image_y": "robot_z", "depth": "robot_y"}
HAND_WRIST_ROLL_SCALE = 1.0
HAND_WRIST_PITCH_SCALE = 1.0
HAND_WRIST_YAW_SCALE = 1.0
HAND_WRIST_ROLL_OFFSET_RAD = 0.0
HAND_WRIST_PITCH_OFFSET_RAD = 0.0
HAND_WRIST_YAW_OFFSET_RAD = 0.0
HAND_WRIST_MAX_ROLL_RAD = WRIST_ROLL_MAX
HAND_WRIST_MAX_PITCH_RAD = WRIST_PITCH_MAX
HAND_WRIST_MAX_YAW_RAD = WRIST_YAW_MAX
HAND_DEPTH_FROM_SIZE_ENABLED = True
HAND_DEPTH_SIZE_SOURCE = "auto"
HAND_DEPTH_MIN_NORM = 0.0
HAND_DEPTH_MAX_NORM = 1.0
HAND_SIZE_NEAR_PIXELS = None
HAND_SIZE_FAR_PIXELS = None
HAND_SIZE_NEAR_NORM = 0.32
HAND_SIZE_FAR_NORM = 0.12
HAND_SIZE_NEAR = HAND_SIZE_NEAR_NORM
HAND_SIZE_FAR = HAND_SIZE_FAR_NORM
HAND_SIZE_DEPTH_INVERT = False
HAND_ARUCO_SIZE_DEPTH_ENABLED = True
HAND_ARUCO_MARKER_SIZE_M = 0.04
HAND_ARUCO_DEPTH_MIN_M = 0.15
HAND_ARUCO_DEPTH_MAX_M = 0.70
HAND_DEPTH_TARGET_NEAR_M = None
HAND_DEPTH_TARGET_FAR_M = None
HAND_DEPTH_AXIS = "robot_y"
# Override early low iteration defaults for the 7-DOF DLS solver.
IK_MAX_ITERS = 80
IK_DLS_MAX_ITERS = 80
IK_DLS_MAX_STEP_RAD = 0.18
# Position-first weighting for tabletop pickup/hand tracking.
IK_DLS_POSITION_GAIN = 6.0
IK_DLS_ORIENTATION_GAIN = 0.35
IK_COST_POSITION = 120.0
IK_COST_ORIENTATION = 0.20
# ---- Main-loop integration ----
# main.py is the only interactive application entry point. Calibration scripts
# remain runnable on their own.
HANDTRACKING_CAMERA_INDEX = 0
HANDTRACKING_CAMERA_CANDIDATE_INDICES = [0, 1, 2]
HANDTRACKING_CAMERA_BACKEND = "default"
HANDTRACKING_FLIP_CAMERA_FRAME = True
MAIN_READ_ROBOT_FEEDBACK = True
PICKPLACE_TRIGGER_KEY = "p"
PICKPLACE_TRIGGER_ON_SNAP = True
# Triggered pick-and-place should complete one pick/place cycle and then stop.
# Set this True only if you intentionally want repeated autonomous cycles.
PICKPLACE_TRIGGER_CONTINUOUS_MODE = False
PICKPLACE_WINDOW_NAME = "Pick and Place"

# ---- Main-loop / handtracking performance safety ----
# These are intentionally placed at the end so they override earlier defaults.
MAIN_LOOP_HZ = 30.0
MAIN_CAMERA_FRAME_WIDTH = 640
MAIN_CAMERA_FRAME_HEIGHT = 480
MAIN_CAMERA_FPS = 30
MAIN_CAMERA_BUFFERSIZE = 1
MAIN_CAMERA_OPEN_RETRIES = 3
MAIN_CAMERA_READ_RETRIES = 30
MAIN_CAMERA_READ_RETRY_DELAY_S = 0.05
MAIN_CAMERA_WARMUP_FRAMES = 5
MAIN_CAMERA_TRY_WITHOUT_PROP_SET_ON_FAIL = True
MAIN_CAMERA_FAIL_FATAL = True
MAIN_CAMERA_VERBOSE_PROBE = True
MAIN_CAMERA_STABILITY_FRAMES = 10
MAIN_CAMERA_STABILITY_TIMEOUT_S = 3.0
MAIN_CAMERA_REOPEN_ON_LIVE_FAILURE = True
MAIN_CAMERA_MAX_REOPEN_ATTEMPTS = 2
MAIN_ROBOT_FEEDBACK_HZ = 5.0
HANDTRACKING_INIT_MEDIAPIPE_BEFORE_CAMERA = True
LOW_LATENCY_MODE = True
CAMERA_THREADED_CAPTURE = True
CAMERA_DROP_OLD_FRAMES = True
CAMERA_MAX_QUEUE_SIZE = 1
CAMERA_CAPTURE_WIDTH = 640
CAMERA_CAPTURE_HEIGHT = 480
CAMERA_CAPTURE_FPS = 30
CAMERA_FORCE_MJPEG = False
HANDTRACKING_CAMERA_USE_NATIVE_VIEW = True
HANDTRACKING_CAMERA_FORCE_RESOLUTION = False
HANDTRACKING_CAMERA_PRINT_FRAME_SHAPE = True
CAMERA_FLUSH_STALE_FRAMES = True
CAMERA_FLUSH_COUNT = 3
HANDTRACKING_PROCESS_EVERY_N_FRAMES = 1
HANDTRACKING_MAX_PROCESS_HZ = 30.0
HANDTRACKING_MEDIAPIPE_MODEL_COMPLEXITY = 0
HANDTRACKING_MIN_DETECTION_CONFIDENCE = 0.55
HANDTRACKING_MIN_TRACKING_CONFIDENCE = 0.55
IK_MAX_SOLVE_HZ = 20.0
IK_REUSE_LAST_ON_RATE_LIMIT = True
IK_TIMEOUT_MS = 20
IK_MAX_ITERS_LOW_LATENCY = 20
ROBOT_COMMAND_MAX_HZ = 25.0
ROBOT_COMMAND_DROP_OLD = True
ROBOT_COMMAND_ASYNC_ONLY = True
DEBUG_PERF_OVERLAY = True
DEBUG_PERF_PRINT_EVERY_S = 2.0
DEBUG_DISABLE_EXPENSIVE_OVERLAYS = False
HAND_TARGET_SMOOTHING_ALPHA_LOW_LATENCY = 0.45
HAND_DEPTH_SMOOTHING_ALPHA_LOW_LATENCY = 0.35
HAND_WRIST_SMOOTHING_ALPHA_LOW_LATENCY = 0.45
HAND_DEPTH_MODE = "monocular_size"
HAND_DEPTH_ENABLE_ARUCO_GLOVE = False
HAND_DEPTH_CAMERA_ENABLED = False
HAND_CNN_DEPTH_ENABLED = False
HAND_ALT_POSE_BACKEND_ENABLED = False
HAND_MONOCULAR_DEPTH_ENABLED = True
HAND_MONOCULAR_DEPTH_CALIBRATION_FILE = "calibration_data/hand_depth_calibration.json"
HAND_MONOCULAR_DEPTH_AUTO_CALIBRATE = False
HAND_MONOCULAR_NEAR_M = 0.20
HAND_MONOCULAR_CENTER_M = 0.45
HAND_MONOCULAR_FAR_M = 0.70
HAND_MONOCULAR_NEAR_SIZE_NORM = 0.32
HAND_MONOCULAR_CENTER_SIZE_NORM = 0.20
HAND_MONOCULAR_FAR_SIZE_NORM = 0.12
HAND_MONOCULAR_REAL_PALM_WIDTH_M = 0.085
HAND_MONOCULAR_REAL_WRIST_TO_MIDDLE_MCP_M = 0.095
HAND_MONOCULAR_USE_CAMERA_INTRINSICS = True
HAND_MONOCULAR_SIZE_FUSION = "median"
HAND_MEDIAPIPE_RELATIVE_Z_ENABLED = True
HAND_MEDIAPIPE_RELATIVE_Z_WEIGHT = 0.12
HAND_MEDIAPIPE_RELATIVE_Z_MAX_EFFECT_M = 0.035
HAND_DEPTH_SMOOTHING_ALPHA = 0.35
HAND_DEPTH_MAX_STEP_M = 0.06
HAND_DEPTH_HOLD_LAST_ON_INVALID = True
HAND_DEPTH_DEFAULT_M = 0.45
HAND_DEPTH_CONFIDENCE_MIN = 0.35
HAND_DEPTH_OUTLIER_REJECT_STD = 2.5
HAND_DEPTH_DEBUG = True
HAND_DEPTH_SHOW_SOURCE = True
HAND_DEPTH_CALIBRATION_SAMPLES_PER_POSE = 40
HAND_DEPTH_CALIBRATION_REQUIRED_STABLE_FRAMES = 20
HAND_DEPTH_CALIBRATION_STABILITY_STD_MAX = 0.015
HAND_DEPTH_CALIBRATION_POSES = ["near", "center", "far"]

# MediaPipe model_complexity=0 is much faster and is usually stable enough for
# real-time robot teleop. Increase to 1 only if landmark quality is too low.
HANDTRACKING_MAX_NUM_HANDS = 2

# Full 7-DOF IK is expensive; solve it at a controlled rate and reuse the last
# solution between solves so the camera preview does not freeze.
HAND_IK_HZ = 6.0
HAND_IK_FORCE_SOLVE_TARGET_DELTA_M = 0.025
REAL_ROBOT_WATCHDOG_FEEDBACK_HZ = 5.0
# ---- Low-latency live handtracking/teleop overrides ----
# Keep camera, IK, and serial I/O decoupled.  The foreground loop should never
# block on a full IK solve or eight per-motor serial writes.
HAND_IK_ASYNC = True
HAND_IK_HZ = IK_MAX_SOLVE_HZ
IK_TELEOP_DLS_MAX_ITERS = IK_MAX_ITERS_LOW_LATENCY
IK_FAST_PREVIOUS_ONLY = True
IK_FAST_INCLUDE_GEOMETRIC_SEED = True
IK_DLS_MAX_ITERS = IK_MAX_ITERS_LOW_LATENCY
IK_DLS_ORIENTATION_GAIN = 0.25
HANDTRACKING_MODEL_COMPLEXITY = HANDTRACKING_MEDIAPIPE_MODEL_COMPLEXITY
HANDTRACKING_MIN_DETECTION_CONFIDENCE = 0.55
HANDTRACKING_MIN_TRACKING_CONFIDENCE = 0.55

REAL_ROBOT_ASYNC_COMMAND_SENDER = True
REAL_ROBOT_VERBOSE_ACTION_LOG = False
DIRECT_FEETECH_USE_SYNC_WRITE = True
DIRECT_FEETECH_INTER_WRITE_DELAY_S = 0.0005
DIRECT_FEETECH_READ_TIMEOUT_S = 0.025

# Robot feedback is useful for initial seeding/watchdog, but reading every joint
# over the serial bus is slow.  Keep it sparse during live handtracking.
MAIN_ROBOT_FEEDBACK_HZ = 1.0
REAL_ROBOT_WATCHDOG_FEEDBACK_HZ = 1.0
# ---- Robust 7-DOF IK improvements ----
# The nominal chain works now. When CAD/URDF-derived calibration is available,
# export a JSON artifact here with link lengths, joint axes, axis signs, zero
# offsets, and collision radius. mathmodel.py will auto-load it.
IK_MODEL_CALIBRATION_FILE = "calibration_artifacts/robot_model_calibration.json"
IK_DEFAULT_MODE = "auto"
IK_JOINT_AXES_LOCAL = ["z", "y", "y", "y", "z", "x", "y"]
IK_COLLISION_CAPSULE_RADIUS_M = 0.018
IK_COLLISION_FLOOR_Z_M = 0.0

# Live teleop IK: prioritizes low latency, continuity, and no sudden jumps.
IK_TELEOP_DLS_MAX_ITERS = 20
IK_TELEOP_DLS_DAMPING = 0.09
IK_TELEOP_DLS_MAX_STEP_RAD = 0.16
IK_TELEOP_DLS_STEP_GAIN = 0.80
IK_TELEOP_POSITION_TOL_M = 0.003
IK_TELEOP_ORIENTATION_TOL_RAD = 0.035
IK_TELEOP_POSITION_GAIN = 6.0
IK_TELEOP_ORIENTATION_GAIN = 0.10
IK_TELEOP_COST_POSITION = 120.0
IK_TELEOP_COST_ORIENTATION = 0.06
IK_TELEOP_COST_CONTINUITY = 0.12
IK_TELEOP_COST_LIMIT = 0.030
IK_TELEOP_COST_POSTURE = 0.010
IK_TELEOP_COST_ELBOW = 0.010
IK_TELEOP_COST_WRIST = 0.010
IK_TELEOP_COST_COLLISION = 0.0
IK_TELEOP_COST_MANIPULABILITY = 0.0
IK_TELEOP_NULLSPACE_LIMIT_GAIN = 0.030
IK_TELEOP_NULLSPACE_CONTINUITY_GAIN = 0.080
IK_TELEOP_NULLSPACE_POSTURE_GAIN = 0.020
IK_TELEOP_NULLSPACE_ELBOW_GAIN = 0.000
IK_TELEOP_NULLSPACE_WRIST_GAIN = 0.004
IK_TELEOP_NULLSPACE_COLLISION_GAIN = 0.0
IK_TELEOP_NULLSPACE_MANIPULABILITY_GAIN = 0.0
IK_TELEOP_STRICT_REACHABILITY = False

# Pick/place or scripted planning IK: tries more seeds and is stricter about
# final Cartesian error. This is slower but better for autonomous waypoints.
IK_PLANNING_DLS_MAX_ITERS = 90
IK_PLANNING_DLS_DAMPING = 0.065
IK_PLANNING_DLS_MAX_STEP_RAD = 0.18
IK_PLANNING_DLS_STEP_GAIN = 0.85
IK_PLANNING_POSITION_TOL_M = 0.0008
IK_PLANNING_ORIENTATION_TOL_RAD = 0.008
IK_PLANNING_POSITION_GAIN = 7.0
IK_PLANNING_ORIENTATION_GAIN = 0.45
IK_PLANNING_COST_POSITION = 160.0
IK_PLANNING_COST_ORIENTATION = 0.35
IK_PLANNING_COST_CONTINUITY = 0.040
IK_PLANNING_COST_LIMIT = 0.050
IK_PLANNING_COST_POSTURE = 0.015
IK_PLANNING_COST_ELBOW = 0.030
IK_PLANNING_COST_WRIST = 0.012
IK_PLANNING_COST_COLLISION = 0.0
IK_PLANNING_COST_MANIPULABILITY = 0.0002
IK_PLANNING_NULLSPACE_LIMIT_GAIN = 0.045
IK_PLANNING_NULLSPACE_CONTINUITY_GAIN = 0.020
IK_PLANNING_NULLSPACE_POSTURE_GAIN = 0.025
IK_PLANNING_NULLSPACE_ELBOW_GAIN = 0.002
IK_PLANNING_NULLSPACE_WRIST_GAIN = 0.006
IK_PLANNING_NULLSPACE_COLLISION_GAIN = 0.0
IK_PLANNING_NULLSPACE_MANIPULABILITY_GAIN = 0.0003
IK_PLANNING_STRICT_REACHABILITY = True

# Secondary redundancy shaping. These can be tuned after the real CAD model is
# available. Defaults are intentionally mild to avoid fighting the primary pose.
IK_ENABLE_ELBOW_REGION_COST = True
IK_ELBOW_MIN_Z_M = 0.015
IK_ELBOW_MIN_RADIAL_M = 0.030
IK_WRIST_AWKWARDNESS_WEIGHTS = [0.0, 0.0, 0.0, 0.25, 0.45, 0.25, 0.45]
IK_COST_ELBOW = 0.02
IK_COST_WRIST = 0.005
IK_COST_MANIPULABILITY = 0.0
IK_NULLSPACE_ELBOW_GAIN = 0.0
IK_NULLSPACE_WRIST_GAIN = 0.004
IK_NULLSPACE_MANIPULABILITY_GAIN = 0.0
IK_NULLSPACE_FD_EPS_RAD = 1e-4

# ---------------------------------------------------------------------------
# Real robot torque/current split and optional outer PID correction
# ---------------------------------------------------------------------------

# Keep torque/current limiting enabled for OCP protection.
REAL_ROBOT_ENABLE_TORQUE_LIMIT = True

# Default fallback torque/current limit if a motor is not listed in a group.
REAL_ROBOT_TORQUE_LIMIT_PERCENT = 100.0

# Per-motor torque/current limit overrides (percent of full PWM, 0-100). Take
# precedence over REAL_ROBOT_TORQUE_LIMIT_GROUPS below. Allocated proportional
# to gravity-hold load: shoulder_lift and elbow_flex carry the most weight,
# wrist motors are small. Sum of stall current at these limits stays well under
# a 10 A rail at 7.4-7.7 V.
REAL_ROBOT_TORQUE_LIMIT_BY_MOTOR_ID = {
    1: 100.0,  # shoulder_pan
    2: 100.0,  # shoulder_lift
    3: 100.0,  # elbow_flex
    4: 100.0,  # wrist_flex
    5: 100.0,   # wrist_yaw
    6: 100.0,   # wrist_roll
    7: 100.0,   # wrist_pitch
    8: 100.0,   # gripper
}

# Floor applied at bus startup before the per-motor table lands. Even if the
# per-motor write fails, motors will not come online unrestricted.
REAL_ROBOT_SAFE_INITIAL_TORQUE_PERCENT = 100.0

# Split available current/torque budget by motor group:
# Motors 1-4 are shoulder/elbow/high-load joints and receive 75%.
# Motors 5-8 are wrist/gripper/lower-load joints and receive 25%.
REAL_ROBOT_TORQUE_LIMIT_GROUPS = {
    "high_load": {
        "motor_ids": [1, 2, 3, 4],
        "percent": 100.0,
    },
    "low_load": {
        "motor_ids": [5, 6, 7, 8],
        "percent": 100.0,
    },
}
REAL_ROBOT_TORQUE_LIMIT_HIGH_LOAD_PERCENT = REAL_ROBOT_TORQUE_LIMIT_GROUPS["high_load"]["percent"]
REAL_ROBOT_TORQUE_LIMIT_LOW_LOAD_PERCENT = REAL_ROBOT_TORQUE_LIMIT_GROUPS["low_load"]["percent"]

# Feetech/ST3215 runtime torque/current-limit register.
# Common STS/SMS current/torque limit scale is 0..1000.
FEETECH_TORQUE_LIMIT_ADDR = 48
FEETECH_TORQUE_LIMIT_RAW_MAX = 1000

# Optional outer joint-position PID.
# This is disabled by default because the servo already has internal PID.
# Enable only after confirming feedback reads are stable and not causing lag.
REAL_ROBOT_PID_ENABLED = False

# PID runs on top of commanded joint targets using measured joint feedback.
# Gains are per joint in order:
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, wrist_pitch, gripper
REAL_ROBOT_PID_KP = [0.08, 0.08, 0.08, 0.06, 0.04, 0.04, 0.04, 0.00]
REAL_ROBOT_PID_KI = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
REAL_ROBOT_PID_KD = [0.01, 0.01, 0.01, 0.008, 0.004, 0.004, 0.004, 0.00]

# Safety clamps for the outer PID correction.
REAL_ROBOT_PID_MAX_CORRECTION_DEG = 2.0
REAL_ROBOT_PID_DEADBAND_DEG = 5
REAL_ROBOT_PID_INTEGRAL_LIMIT_DEG_S = 8.0
REAL_ROBOT_PID_RESET_ON_TARGET_JUMP_DEG = 8.0
REAL_ROBOT_PID_VERBOSE = False
REAL_ROBOT_ENABLE_OUTER_PID = REAL_ROBOT_PID_ENABLED
REAL_ROBOT_PID_DERIVATIVE_FILTER_ALPHA = 0.25
REAL_ROBOT_PID_APPLY_TO_GRIPPER = False
REAL_ROBOT_FEEDBACK_CACHE_S = 0.10

# ---------------------------------------------------------------------------
# Final max-responsiveness / low-latency command overrides
# ---------------------------------------------------------------------------
# These final assignments intentionally override all earlier duplicates in this
# file. Motor torque/current remains commanded at the Feetech register maximum,
# while software velocity/acceleration/step limits, smoothing delay, IK rate
# limits, command-rate limits, and serial write delays are pushed to the most
# responsive practical settings for live gesture teleoperation.

# Motor-side software limits: effectively inactive.
REAL_ROBOT_ENABLE_TORQUE_LIMIT = True
REAL_ROBOT_MAX_VELOCITY_DEG = 1000000.0
REAL_ROBOT_MAX_ACCELERATION_DEG = 1000000.0
REAL_ROBOT_GRIPPER_MAX_VELOCITY_DEG = 1000000.0
REAL_ROBOT_GRIPPER_MAX_ACCELERATION_DEG = 1000000.0
REAL_ROBOT_MAX_RELATIVE_TARGET_DEG = 360.0
REAL_ROBOT_ACTION_DEADBAND_DEG = 0.0
REAL_ROBOT_TORQUE_LIMIT_PERCENT = 100.0
REAL_ROBOT_SAFE_INITIAL_TORQUE_PERCENT = 100.0
REAL_ROBOT_TORQUE_LIMIT_BY_MOTOR_ID = {
    1: 100.0,  # shoulder_pan
    2: 100.0,  # shoulder_lift
    3: 100.0,  # elbow_flex
    4: 100.0,  # wrist_flex
    5: 100.0,  # wrist_yaw
    6: 100.0,  # wrist_roll
    7: 100.0,  # wrist_pitch
    8: 100.0,  # gripper
}
REAL_ROBOT_TORQUE_LIMIT_GROUPS = {
    "high_load": {"motor_ids": [1, 2, 3, 4], "percent": 100.0},
    "low_load": {"motor_ids": [5, 6, 7, 8], "percent": 100.0},
}
FEETECH_TORQUE_LIMIT_RAW_MAX = 1000

# Runtime command-rate limits: high enough that camera/IK/serial throughput are
# the limiting factors instead of this config file.
REAL_ROBOT_HZ = 100.0
ROBOT_COMMAND_MAX_HZ = 100.0
MAIN_LOOP_HZ = 60.0
HANDTRACKING_MAX_PROCESS_HZ = 60.0
IK_MAX_SOLVE_HZ = 60.0
HAND_IK_HZ = 60.0
MAIN_CAMERA_FPS = 60
CAMERA_CAPTURE_FPS = 60

# Always use the newest available command/frame; never build a backlog.
LOW_LATENCY_MODE = True
CAMERA_THREADED_CAPTURE = True
CAMERA_DROP_OLD_FRAMES = True
CAMERA_MAX_QUEUE_SIZE = 1
CAMERA_FLUSH_STALE_FRAMES = True
CAMERA_FLUSH_COUNT = 3
HANDTRACKING_PROCESS_EVERY_N_FRAMES = 1
ROBOT_COMMAND_DROP_OLD = True
ROBOT_COMMAND_ASYNC_ONLY = True
REAL_ROBOT_ASYNC_COMMAND_SENDER = True
DIRECT_FEETECH_USE_SYNC_WRITE = True

# Remove artificial smoothing/lag in live teleop. Alpha=1.0 means the newest
# target replaces the old target immediately for standard EMA-style filters.
JOINT_SMOOTH_ALPHA = 1.0
ANGLE_SMOOTH_ALPHA = 1.0
HAND_CMD_SMOOTHING = 1.0
POSE_SMOOTH_ALPHA = 1.0
HAND_STATE_SMOOTHING = 1.0
HAND_TARGET_SMOOTHING_ALPHA_LOW_LATENCY = 1.0
HAND_DEPTH_SMOOTHING_ALPHA_LOW_LATENCY = 1.0
HAND_WRIST_SMOOTHING_ALPHA_LOW_LATENCY = 1.0
HAND_DEPTH_SMOOTHING_ALPHA = 1.0
HAND_DEPTH_AUTOCALIBRATE_EMA_ALPHA = 1.0

# Make gesture transitions immediate rather than cooldown-limited.
CLAP_COOLDOWN_S = 0.0
SNAP_COOLDOWN_S = 0.0

# IK responsiveness: solve more often, allow larger steps, and do not reuse old
# IK solely because of rate limiting. If the implementation supports a timeout,
# keep it finite so one bad solve cannot freeze the camera loop.
IK_REUSE_LAST_ON_RATE_LIMIT = False
IK_TIMEOUT_MS = 100
IK_MAX_ITERS_LOW_LATENCY = 60
IK_MAX_ITERS = 100
IK_DLS_MAX_ITERS = 60
IK_TELEOP_DLS_MAX_ITERS = 60
IK_DLS_MAX_STEP_RAD = 3.141592653589793
IK_TELEOP_DLS_MAX_STEP_RAD = 3.141592653589793
IK_DLS_STEP_GAIN = 1.0
IK_TELEOP_DLS_STEP_GAIN = 1.0
IK_FAST_PREVIOUS_ONLY = False
IK_FAST_INCLUDE_GEOMETRIC_SEED = True
HAND_IK_ASYNC = True
HAND_IK_FORCE_SOLVE_TARGET_DELTA_M = 0.0

# Serial/runtime feedback reads can block command writes on a half-duplex bus.
# Disable live feedback reads for minimum gesture-to-motion delay; calibration
# and explicit diagnostics still use their own read settings above.
MAIN_READ_ROBOT_FEEDBACK = False
MAIN_ROBOT_FEEDBACK_HZ = 0.2
REAL_ROBOT_WATCHDOG_FEEDBACK_HZ = 0.2
REAL_ROBOT_ENABLE_TRACKING_WATCHDOG = False
DIRECT_FEETECH_INTER_WRITE_DELAY_S = 0.0
DIRECT_FEETECH_READ_TIMEOUT_S = 0.005
REAL_ROBOT_VERBOSE_ACTION_LOG = False
DEBUG_PERF_OVERLAY = False
DEBUG_DISABLE_EXPENSIVE_OVERLAYS = True

# If the optional outer PID is enabled later, remove its software correction
# clamps/deadband. It remains disabled here unless robot_controller.py enables it.
REAL_ROBOT_PID_ENABLED = False
REAL_ROBOT_PID_MAX_CORRECTION_DEG = 360.0
REAL_ROBOT_PID_DEADBAND_DEG = 0.0
REAL_ROBOT_PID_INTEGRAL_LIMIT_DEG_S = 1000000.0
REAL_ROBOT_PID_RESET_ON_TARGET_JUMP_DEG = 360.0

# ---------------------------------------------------------------------------
# Hand workspace learning and virtual wrist orientation
# ---------------------------------------------------------------------------
# This calibration complements, and never replaces, robot_joint_calibration.json.
HAND_WORKSPACE_CALIBRATION_FILE = "calibration_data/hand_workspace_calibration.json"
HAND_WORKSPACE_LEARNING_ENABLED = True
HAND_WORKSPACE_MAPPING_METHOD = "rbf_residual"  # piecewise_affine | thin_plate_rbf | rbf_residual | knn_weighted
HAND_WORKSPACE_FALLBACK_METHOD = "piecewise_affine"
HAND_WORKSPACE_RESIDUAL_MAX_M = 0.030
HAND_WORKSPACE_MIN_EXAMPLES_FOR_RBF = 8
HAND_WORKSPACE_RBF_KERNEL = "thin_plate"  # thin_plate | multiquadric | gaussian
HAND_WORKSPACE_RBF_SMOOTHING = 1e-4
HAND_WORKSPACE_KNN_K = 4
HAND_WORKSPACE_USE_JOINT_SEED_EXAMPLES = True
HAND_WORKSPACE_DIRECT_JOINT_LEARNING_ENABLED = False
HAND_WORKSPACE_CAPTURE_POSES = [
    "center",
    "max_left",
    "max_right",
    "max_up",
    "max_down",
    "max_near",
    "max_far",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "near_left",
    "near_right",
    "far_left",
    "far_right",
]

# Robot-side workspace extrema calibration. robot_calibrate.py records only
# robot readback/FK poses; handtracking.py supplies live RGB/MediaPipe hand
# inputs and IK remains the final joint-command generator.
ROBOT_WORKSPACE_CALIBRATION_FILE = "calibration_data/robot_workspace_extrema_calibration.json"
ROBOT_WORKSPACE_LEGACY_MIRROR_CALIBRATION_FILE = "calibration_data/robot_mirror_workspace_calibration.json"
ROBOT_WORKSPACE_ENABLED = True
ROBOT_WORKSPACE_REQUIRED_POSES = [
    "center",
    "left",
    "right",
    "up",
    "down",
    "near",
    "far",
]
ROBOT_WORKSPACE_OPTIONAL_POSES = [
    "up_left",
    "up_right",
    "down_left",
    "down_right",
    "near_left",
    "near_right",
    "far_left",
    "far_right",
    "near_up",
    "near_down",
    "far_up",
    "far_down",
    "near_up_left",
    "near_up_right",
    "far_down_left",
    "far_down_right",
]
ROBOT_WORKSPACE_MAPPING_METHOD = "axis_vector_knn_residual"  # axis_vector | axis_vector_knn_residual | axis_vector_rbf_residual
ROBOT_WORKSPACE_KNN_ENABLED = True
ROBOT_WORKSPACE_KNN_K = 4
ROBOT_WORKSPACE_RBF_ENABLED = True
ROBOT_WORKSPACE_RBF_MIN_SAMPLES = 8
ROBOT_WORKSPACE_RBF_KERNEL = "thin_plate"  # thin_plate | gaussian | multiquadric
ROBOT_WORKSPACE_RBF_SMOOTHING = 1e-4
ROBOT_WORKSPACE_RESIDUAL_MAX_M = 0.030
ROBOT_WORKSPACE_CLAMP_TO_RECORDED_BOUNDS = True
ROBOT_WORKSPACE_CLAMP_MARGIN_M = 0.020
ROBOT_WORKSPACE_PROJECT_TO_CENTER_ON_IK_FAIL = True
ROBOT_WORKSPACE_PROJECTION_STEPS = 8
ROBOT_WORKSPACE_USE_JOINT_SEED_EXAMPLES = True
ROBOT_WORKSPACE_DIRECT_JOINT_LEARNING_ENABLED = False

HAND_WORKSPACE_HORIZONTAL_FLIP = False
HAND_WORKSPACE_VERTICAL_FLIP = True
HAND_WORKSPACE_DEPTH_FLIP = False
HAND_WORKSPACE_CENTER_X_NORM = 0.5
HAND_WORKSPACE_CENTER_Y_NORM = 0.5
HAND_WORKSPACE_CENTER_DEPTH_NORM = 0.5
HAND_WORKSPACE_CLAMP_INPUTS = True

# Backward-compatible mirror names. The old JSON file is still supported by
# RobotWorkspaceMapper as a legacy input, but new calibration writes the
# canonical robot_workspace_extrema_calibration.json file.
ROBOT_MIRROR_WORKSPACE_CALIBRATION_FILE = ROBOT_WORKSPACE_CALIBRATION_FILE
HAND_MIRROR_POSITION_CALIBRATION_FILE = "calibration_data/hand_mirror_position_calibration.json"
ROBOT_MIRROR_PAIRED_CALIBRATION_ENABLED = True
ROBOT_MIRROR_WORKSPACE_ENABLED = ROBOT_WORKSPACE_ENABLED
ROBOT_MIRROR_MAPPING_METHOD = "paired_axis_blend_knn_residual"  # axis_blend | paired_axis_blend | paired_axis_blend_knn_residual | paired_axis_blend_rbf_residual
ROBOT_MIRROR_FALLBACK_METHOD = "axis_blend"
ROBOT_MIRROR_DEFAULT_RESIDUAL_METHOD = "knn"
ROBOT_MIRROR_KNN_ENABLED = True
ROBOT_MIRROR_KNN_K = 4
ROBOT_MIRROR_RBF_ENABLED = True
ROBOT_MIRROR_RBF_MIN_SAMPLES = 8
ROBOT_MIRROR_RBF_KERNEL = "thin_plate"  # thin_plate | gaussian | multiquadric
ROBOT_MIRROR_RBF_SMOOTHING = 1e-4
ROBOT_MIRROR_RESIDUAL_MAX_M = 0.030
ROBOT_MIRROR_CLAMP_TO_CALIBRATED_BOUNDS = True
ROBOT_MIRROR_CLAMP_MARGIN_M = 0.020
ROBOT_MIRROR_ANCHOR_WARN_ERR_M = 0.015
ROBOT_MIRROR_ANCHOR_CRITICAL_ERR_M = 0.030
ROBOT_MIRROR_AUDIT_REPORT_FILE = "calibration_data/mirror_calibration_audit.md"
ROBOT_MIRROR_AUDIT_JSON_FILE = "calibration_data/mirror_calibration_audit.json"
ROBOT_MIRROR_USE_JOINT_SEED_EXAMPLES = True
ROBOT_MIRROR_DIRECT_JOINT_LEARNING_ENABLED = False

HAND_MIRROR_CALIBRATION_USE_RUNTIME_FRAME_PREPROCESSING = True
HAND_MIRROR_DEPTH_PAIRING_SOURCE = "raw"  # raw | filtered | hand_size
HAND_MIRROR_HORIZONTAL_FLIP = False
HAND_MIRROR_VERTICAL_FLIP = True
HAND_MIRROR_DEPTH_FLIP = False
HAND_MIRROR_CENTER_X_NORM = 0.5
HAND_MIRROR_CENTER_Y_NORM = 0.5
HAND_MIRROR_CENTER_DEPTH_NORM = 0.5
HAND_MIRROR_CLAMP_INPUTS = True
# Paired hand mirror calibration maps camera hand extrema to matching robot pose names.
# If the hand calibration file is missing, mapper falls back to ideal normalized extrema.
ROBOT_MIRROR_REQUIRED_POSES = [
    "center",
    "mirror_left",
    "mirror_right",
    "mirror_up",
    "mirror_down",
    "mirror_near",
    "mirror_far",
]
ROBOT_MIRROR_OPTIONAL_POSES = [
    "mirror_up_left",
    "mirror_up_right",
    "mirror_down_left",
    "mirror_down_right",
    "mirror_near_left",
    "mirror_near_right",
    "mirror_far_left",
    "mirror_far_right",
    "mirror_near_up",
    "mirror_near_down",
    "mirror_far_up",
    "mirror_far_down",
    "mirror_near_up_left",
    "mirror_near_up_right",
    "mirror_far_down_left",
    "mirror_far_down_right",
]

HAND_VIRTUAL_WRIST_ENABLED = True
HAND_VIRTUAL_WRIST_USE_WORLD_LANDMARKS = True
HAND_VIRTUAL_WRIST_ORIENTATION_SMOOTHING = 0.45
HAND_VIRTUAL_WRIST_CONFIDENCE_MIN = 0.35
HAND_VIRTUAL_WRIST_BLEND_TO_NEUTRAL_ON_LOW_CONF = True
HAND_VIRTUAL_WRIST_MAX_ROLL_RAD = WRIST_ROLL_MAX
HAND_VIRTUAL_WRIST_MAX_PITCH_RAD = WRIST_PITCH_MAX
HAND_VIRTUAL_WRIST_MAX_YAW_RAD = WRIST_YAW_MAX

# Simple hand-roll override copied from simple_main.py's useful 5-DOF behavior.
# Motor 7 is exposed as wrist_pitch in JointCommand, but physically behaves like
# a roll axis on this build.
HAND_SIMPLE_PALM_ROLL_ENABLED = True
HAND_SIMPLE_PALM_ROLL_TARGET_JOINT = "wrist_pitch"
HAND_SIMPLE_PALM_ROLL_LEFT_RAD = -0.6
HAND_SIMPLE_PALM_ROLL_RIGHT_RAD = 0.6
HAND_SIMPLE_PALM_ROLL_OUTPUT_LEFT_RAD = -1.5
HAND_SIMPLE_PALM_ROLL_OUTPUT_RIGHT_RAD = 1.5
HAND_SIMPLE_PALM_ROLL_SMOOTHING = 0.60
HAND_SIMPLE_PALM_ROLL_CLAMP = True
HAND_SIMPLE_PALM_ROLL_DEBUG = True

# Backward-compatible aliases for the names used in simple_main.py.
HAND_ROLL_LEFT_NORM = HAND_SIMPLE_PALM_ROLL_LEFT_RAD
HAND_ROLL_RIGHT_NORM = HAND_SIMPLE_PALM_ROLL_RIGHT_RAD
M7_AT_HAND_ROLL_LEFT = HAND_SIMPLE_PALM_ROLL_OUTPUT_LEFT_RAD
M7_AT_HAND_ROLL_RIGHT = HAND_SIMPLE_PALM_ROLL_OUTPUT_RIGHT_RAD


# Low-cost extension shaping.  Horizontal response is intentionally identity
# because current left/right tracking is the known-good baseline.
ROBOT_WORKSPACE_HORIZONTAL_RESPONSE_GAMMA = 1.0
ROBOT_WORKSPACE_VERTICAL_RESPONSE_GAMMA = 0.85
ROBOT_WORKSPACE_DEPTH_RESPONSE_GAMMA = 0.85
ROBOT_WORKSPACE_VERTICAL_ENDPOINT_BOOST_ENABLED = True
ROBOT_WORKSPACE_DEPTH_ENDPOINT_BOOST_ENABLED = True
ROBOT_WORKSPACE_EXTENSION_SHAPING_CLAMP = True

# Optional O(1) post-IK wrist-roll drive for motor 6.  This uses the already
# measured image-plane palm roll and does not trigger extra IK solves.
HAND_DIRECT_WRIST_ROLL_ENABLED = True
HAND_DIRECT_WRIST_ROLL_SOURCE = "palm_roll"
HAND_DIRECT_WRIST_ROLL_OUTPUT_LEFT_RAD = -0.45
HAND_DIRECT_WRIST_ROLL_OUTPUT_RIGHT_RAD = 0.45
HAND_DIRECT_WRIST_ROLL_BLEND = 1.0
HAND_DIRECT_WRIST_ROLL_CLAMP = True
HAND_DIRECT_WRIST_ROLL_DEADBAND_RAD = 0.03
HAND_DIRECT_WRIST_ROLL_DEBUG = True

# Prevent robot workspace examples from pinning motor 6 during teleop when a
# previous solution already exists.  The previous wrist roll is a better seed
# than a static calibration pose for continuous hand control.
ROBOT_WORKSPACE_SEED_EXCLUDE_WRIST_ROLL_FOR_TELEOP = True
ROBOT_MIRROR_SEED_EXCLUDE_WRIST_ROLL_FOR_TELEOP = True

# Final speed-percent defaults must remain after all max-responsiveness
# overrides. Set REAL_ROBOT_ARM_SPEED_PERCENT below 100 only when you
# intentionally want motors 1-7 slowed; motor 8 gripper is never scaled.
REAL_ROBOT_BASE_COMMAND_MAX_HZ = ROBOT_COMMAND_MAX_HZ
REAL_ROBOT_BASE_HZ = REAL_ROBOT_HZ
HAND_IK_BASE_HZ = HAND_IK_HZ
IK_MAX_SOLVE_BASE_HZ = IK_MAX_SOLVE_HZ
REAL_ROBOT_ARM_SPEED_PERCENT = 35.0
REAL_ROBOT_MIN_ARM_SPEED_PERCENT = 1.0
REAL_ROBOT_APPLY_SPEED_PERCENT_TO_TORQUE = True
REAL_ROBOT_APPLY_SPEED_PERCENT_TO_RATES = False
REAL_ROBOT_SPEED_PERCENT_VERBOSE = True

# ---------------------------------------------------------------------------
# Hand-tracking runtime overrides (place at end-of-file so they win)
# Diagnostics ON, feedback ON, and force the legacy 4-DOF proportional mapper
# instead of cartesian IK. Cartesian IK is currently picking a redundant-DOF
# branch with shoulder_lift ~ -73 deg (arm into the ground) on roughly half of
# the calibrated mirror-anchor positions and showing IK=pending/fallback every
# frame. Switching to the proportional path (`_old_landmarks_to_command` in
# handtracking.py:1782) keeps the arm tracking hand x/y/depth in 3D with
# pan/lift/elbow/wrist_flex; optional simple palm roll can still drive motor 7
# through the wrist_pitch JointCommand field.
# Re-enable cartesian once the mirror calibration is recomputed against the
# current chain or the unreachable anchors are dropped.
DEBUG_DISABLE_EXPENSIVE_OVERLAYS = False
DEBUG_PERF_OVERLAY = True
DEBUG_PERF_PRINT_EVERY_S = 1.0
HAND_CAMERA_AXIS_DEBUG = True
MAIN_READ_ROBOT_FEEDBACK = True
MAIN_ROBOT_FEEDBACK_HZ = 5.0
REAL_ROBOT_VERBOSE_ACTION_LOG = False
HAND_IK_ASYNC = True
HAND_USE_CARTESIAN_IK = False
HAND_CARTESIAN_MAPPING_ENABLED = False

# Joint command limits used by robot_controller._joint_command_to_action.
# The earlier defaults (ELBOW_MIN=-0.5, SHOULDER_LIFT_MAX=+2.0) were tuned for
# the cartesian IK + extended-arm neutral. With the simple_handtrack pipeline
# and a folded neutral, the IK output sits in the +1.4..+2.1 lift range and
# -0.7..-1.3 elbow range, so the old caps were clamping every command and
# pinning the joints. Widen to the motor's mechanical range; the calibrated
# min_pos/max_pos in robot_joint_calibration.json is the real hard stop.
import math as _m
BASE_PAN_MIN = -_m.pi
BASE_PAN_MAX = _m.pi
SHOULDER_LIFT_MIN = -_m.pi
SHOULDER_LIFT_MAX = _m.pi
ELBOW_MIN = -_m.pi
ELBOW_MAX = _m.pi
WRIST_FLEX_MIN = -_m.pi
WRIST_FLEX_MAX = _m.pi

# Speed/accel are governed entirely by the torque limit below — speed-percent
# scaling and software velocity/accel ramps did not produce a usable feel.
# Keep torque scaling off so the limit reads cleanly from the percent values.
REAL_ROBOT_APPLY_SPEED_PERCENT_TO_TORQUE = False
REAL_ROBOT_ARM_SPEED_PERCENT = 100.0

# Hand-tracking debug: lean on a global torque cap instead of software
# velocity/accel limits. Lower torque -> motors physically can't slam, gives
# a gentler feel and yields to obstacles, while still allowing fast motion
# under low load. Lift (motor 2) needs higher torque than the rest to fight
# gravity; bump it up if you see it stall on lifts.
REAL_ROBOT_TORQUE_LIMIT_PERCENT = 30.0
REAL_ROBOT_TORQUE_LIMIT_BY_MOTOR_ID = {
    1: 30.0,   # shoulder_pan
    2: 45.0,   # shoulder_lift  (gravity load; keep higher to avoid stalls)
    3: 30.0,   # elbow_flex
    4: 30.0,   # wrist_flex
    5: 30.0,   # wrist_yaw
    6: 30.0,   # wrist_roll
    7: 30.0,   # wrist_pitch
    8: 100.0,   # gripper
}