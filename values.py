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

#replace with your own file path
URDF_PATH = "/Users/ricky/PycharmProjects/ECE 3161/SO-ARM100"
