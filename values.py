CAM_W, CAM_H = 720, 720

# Smoothing (0..1)
JOINT_SMOOTH_ALPHA = 0.20
ANGLE_SMOOTH_ALPHA = 0.25

# Physics
SIM_HZ = 240
MOTOR_FORCE = 800
POS_GAIN = 0.35
VEL_GAIN = 1.00

# Log overlay
LOG_MAX = 12
LOG_DURATION = 6.0

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

INVERT_SHOULDER_LIFT = True
INVERT_BASE_PAN = False
INVERT_ELBOW = False
INVERT_WRIST_FLEX = False
INVERT_WRIST_ROLL = False


INVERT_GRIPPER = False

# x: hand left/right on screen
WORKSPACE_X_MIN = -0.18
WORKSPACE_X_MAX =  0.18

# y: depth/reach (farther hand -> larger y)
WORKSPACE_Y_MIN = 0.12
WORKSPACE_Y_MAX = 0.38

# z: hand up/down on screen (up -> larger z)
WORKSPACE_Z_MIN = 0.04
WORKSPACE_Z_MAX = 0.28

# IK controls
IK_MAX_ITERS = 60
IK_RESIDUAL_THRESH = 1e-4

# Smooth target pose a bit (0..1)
POSE_SMOOTH_ALPHA = 0.25

CLOSED_TIPS_ON = 0.060
CLOSED_TIPS_OFF = 0.090

#replace with your own file path
URDF_PATH = "/Users/ricky/PycharmProjects/ECE3161/SO-ARM100"


# Real robot control
ENABLE_REAL_ROBOT = True
REAL_ROBOT_PORT = ""
REAL_ROBOT_HZ = 20.0
REAL_ROBOT_AUTO_CALIBRATE = True
REAL_ROBOT_MAX_RELATIVE_TARGET_DEG = 10
REAL_ROBOT_ACTION_DEADBAND_DEG = 0.5
REAL_ROBOT_JOINT_OFFSETS_DEG = [0, 0, 0, 0, 0]
REAL_ROBOT_ID = "my_awesome_follower_arm"

ARUCO_GLOVE_ENABLED = True
ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_GLOVE_FRONT_ID = 10
ARUCO_GLOVE_BACK_ID = 11
ARUCO_MARKER_SIZE_M = 0.03

CALIB_INTRINSICS_FILE = "calibration_data/calibration_intrinsics.npz"
CALIB_EXTRINSICS_FILE = "calibration_data/calibration_extrinsics.npz"
CALIB_WORKSPACE_FILE = "calibration_data/calibration_workspace.npz"

EXTRINSICS_MODE = "workspace_to_camera"

ARUCO_WORKSPACE_MIN = (-0.18, -0.12, 0.02)
ARUCO_WORKSPACE_MAX = (0.18, 0.18, 0.28)

HAND_CMD_SMOOTHING = 0.25

LEROBOT_CALIBRATION_FILE = ""

# Fine tuning offsets for the physical arm after calibration.
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
REAL_ROBOT_JOINT_OFFSETS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]