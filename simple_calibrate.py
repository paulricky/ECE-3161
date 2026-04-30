"""Simple guided calibration for the SO-100/7-DOF arm.

Captures everything `simple_main.py` needs:
  - per-motor neutral position (motor tick value at the 0-rad pose)
  - per-motor min/max travel range
  - drive_mode = 0 (no inversion), since simple_handtrack.py applies its own
    JOINT_SIGN_* flags if a joint runs the wrong way

Output: calibration_data/robot_joint_calibration.json (the same path the rest
of the project reads). The previous file is backed up to .bak first.

Run:
    python3 simple_calibrate.py

You'll be walked through three steps:
  1. Verify motors respond
  2. Move arm to NEUTRAL pose, press ENTER to record
  3. For each joint, move it through full travel, ENTER when done
"""

from __future__ import annotations

import glob
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import values as val
from robot_controller import _DirectFeetechBus  # noqa: PLC2701  (intentional internal import)


PROJECT_ROOT = Path(__file__).resolve().parent
CALIB_DIR = PROJECT_ROOT / "calibration_data"
CALIB_FILE = CALIB_DIR / "robot_joint_calibration.json"
BAK_FILE = CALIB_DIR / "robot_joint_calibration.json.bak"

MOTOR_NAMES = list(getattr(val, "REAL_ROBOT_MOTOR_NAMES",
    ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
     "wrist_yaw", "wrist_roll", "wrist_pitch", "gripper"]))
MOTOR_IDS = list(getattr(val, "REAL_ROBOT_MOTOR_IDS", [1, 2, 3, 4, 5, 6, 7, 8]))
BAUDRATE = int(getattr(val, "REAL_ROBOT_BAUDRATE", 1000000))
PRESENT_POS_ADDR = int(getattr(val, "FEETECH_PRESENT_POSITION_ADDR", 56))


@dataclass
class MotorRange:
    name: str
    motor_id: int
    neutral: int = 0
    lo: int = 4095
    hi: int = 0


def _auto_detect_port() -> str:
    cfg = str(getattr(val, "REAL_ROBOT_PORT", "")).strip()
    if cfg:
        return cfg
    candidates = []
    for pat in ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/tty.usbmodem*", "/dev/tty.usbserial*"):
        candidates.extend(glob.glob(pat))
    candidates = sorted(dict.fromkeys(candidates))
    if not candidates:
        print("ERROR: no serial ports found. Plug the arm in or set values.REAL_ROBOT_PORT.")
        sys.exit(1)
    if len(candidates) == 1:
        return candidates[0]
    print("Multiple serial ports detected:")
    for i, p in enumerate(candidates):
        print(f"  [{i}] {p}")
    while True:
        choice = input("Pick one [0]: ").strip() or "0"
        try:
            idx = int(choice)
            return candidates[idx]
        except (ValueError, IndexError):
            print("invalid choice")


def _build_bus(port: str) -> _DirectFeetechBus:
    bootstrap_cal = {
        "motor_names": MOTOR_NAMES,
        "motor_ids": MOTOR_IDS,
        "motor_baudrates": [BAUDRATE] * len(MOTOR_IDS),
        # Provide stub calibration so _DirectFeetechBus.connect() doesn't snap motors
        # to stale goals. Real values are written at the end of this script.
        "neutral_pos": [2048] * len(MOTOR_IDS),
        "min_pos": [0] * len(MOTOR_IDS),
        "max_pos": [4095] * len(MOTOR_IDS),
        "homing_offset": [0] * len(MOTOR_IDS),
        "drive_mode": [0] * len(MOTOR_IDS),
    }
    return _DirectFeetechBus(port, bootstrap_cal)


def _read_position(bus: _DirectFeetechBus, motor_id: int) -> Optional[int]:
    val_ = bus._read_u16(int(motor_id), PRESENT_POS_ADDR)
    if val_ is None:
        return None
    return int(val_) & 0xFFFF


def _read_all(bus: _DirectFeetechBus) -> dict[str, Optional[int]]:
    return {name: _read_position(bus, mid) for name, mid in zip(MOTOR_NAMES, MOTOR_IDS, strict=True)}


def _print_positions(positions: dict[str, Optional[int]]) -> None:
    for name, mid in zip(MOTOR_NAMES, MOTOR_IDS, strict=True):
        v = positions.get(name)
        v_str = "----" if v is None else f"{v:4d}"
        print(f"    motor {mid} {name:14s}: {v_str}")


def _verify_motors(bus: _DirectFeetechBus) -> bool:
    print("[1/3] verifying motors")
    print("  reading present position from each motor...")
    positions = _read_all(bus)
    _print_positions(positions)
    missing = [name for name, v in positions.items() if v is None]
    if missing:
        print(f"  FAILED: no response from {missing}")
        print("  Check power, USB cable, and that motor IDs are 1..8 on the bus.")
        return False
    print("  all 8 motors responded.")
    return True


def _capture_neutral(bus: _DirectFeetechBus) -> dict[str, int]:
    print()
    print("[2/3] NEUTRAL pose")
    print("  Disabling torque so you can move the arm by hand.")
    bus.enable_torque(False)
    time.sleep(0.3)
    print()
    print("  Move the arm into its FOLDED / STOWED pose. This is what motor")
    print("  command (0,0,0,0,...) will correspond to. Default folded shape:")
    print()
    print("    - shoulder_pan:    centered (no twist)")
    print("    - shoulder_lift:   upper arm pointing straight UP (vertical)")
    print("    - elbow_flex:      forearm folded back 180 deg, parallel to upper arm")
    print("                       (so forearm points DOWN alongside the upper arm)")
    print("    - wrist_flex:      gripper in line with forearm (also pointing down)")
    print("    - wrist_yaw:       no twist about the forearm axis")
    print("    - wrist_roll:      gripper jaw plane vertical (or your preferred default)")
    print("    - wrist_pitch:     no extra tilt at the gripper")
    print("    - gripper:         jaws fully open")
    print()
    print("  If your folded pose differs from this default, edit NEUTRAL_*_RAD")
    print("  at the top of simple_handtrack.py to match.")
    print()
    input("  Hold the arm in that pose, then press ENTER to record... ")
    positions = _read_all(bus)
    if any(v is None for v in positions.values()):
        print("  read failed; check connections and try again")
        sys.exit(1)
    print("  Recorded neutral positions:")
    _print_positions(positions)
    return {name: int(v) for name, v in positions.items()}


def _capture_ranges(bus: _DirectFeetechBus, neutral: dict[str, int]) -> dict[str, tuple[int, int]]:
    print()
    print("[3/3] JOINT RANGES")
    print("  For each joint, move it through its full travel (one extreme to the")
    print("  other and back). Min/max are recorded continuously while you move.")
    print("  Press ENTER when finished with the joint to advance.")
    print()

    ranges: dict[str, tuple[int, int]] = {}
    for name, mid in zip(MOTOR_NAMES, MOTOR_IDS, strict=True):
        n = neutral[name]
        rng = MotorRange(name=name, motor_id=mid, neutral=n, lo=n, hi=n)
        print(f"  --> Move {name} (motor {mid}) through its FULL range now.")
        print(f"      neutral={n}. Press ENTER when done.")

        # Sample positions in the background of the input() call by polling
        # before reading stdin. We can't truly poll while input() blocks, so
        # instead we sample for a fixed window: the user moves, we sample,
        # then they press ENTER.
        try:
            done = False
            print("      sampling... move now. (any key + ENTER to stop)")
            sys.stdout.flush()
            while not done:
                # Sample several reads quickly between checks for keypress.
                for _ in range(20):
                    v = _read_position(bus, mid)
                    if v is None:
                        continue
                    if v < rng.lo:
                        rng.lo = v
                    if v > rng.hi:
                        rng.hi = v
                    time.sleep(0.01)
                # show current min/max so the user can verify they covered the range
                print(f"        live: min={rng.lo} max={rng.hi}  (ENTER to finish)", end="\r")
                sys.stdout.flush()
                # non-blocking-ish: poll stdin; on Mac/zsh, simplest reliable
                # approach is select.
                import select
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    sys.stdin.readline()
                    done = True
        finally:
            print()
        # Pad the recorded range slightly inward so we don't sit on the hard stop.
        if rng.hi - rng.lo < 10:
            print(f"      WARNING: only {rng.hi - rng.lo} ticks of travel captured; did you move it?")
        print(f"      {name}: min={rng.lo}  max={rng.hi}  span={rng.hi - rng.lo}")
        ranges[name] = (rng.lo, rng.hi)
    return ranges


def _build_calibration_payload(neutral: dict[str, int], ranges: dict[str, tuple[int, int]]) -> dict:
    neutral_list = [neutral[n] for n in MOTOR_NAMES]
    min_list = [ranges[n][0] for n in MOTOR_NAMES]
    max_list = [ranges[n][1] for n in MOTOR_NAMES]
    homing_offset = [-x for x in neutral_list]
    drive_mode = [0] * len(MOTOR_NAMES)
    return {
        "created_at_unix": time.time(),
        "source": "simple_calibrate.py",
        "motor_names": MOTOR_NAMES,
        "motor_ids": MOTOR_IDS,
        "motor_baudrates": [BAUDRATE] * len(MOTOR_IDS),
        "neutral_pos": neutral_list,
        "min_pos": min_list,
        "max_pos": max_list,
        "homing_offset": homing_offset,
        "drive_mode": drive_mode,
    }


def _save(payload: dict) -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    if CALIB_FILE.exists():
        try:
            shutil.copy2(CALIB_FILE, BAK_FILE)
            print(f"  backed up existing calibration -> {BAK_FILE.name}")
        except Exception as exc:
            print(f"  WARNING: could not back up existing calibration: {exc}")
    with CALIB_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {CALIB_FILE}")


def main() -> int:
    print("simple_calibrate.py — guided calibration for the SO-100 7-DOF arm")
    print()
    port = _auto_detect_port()
    print(f"using port: {port}")
    bus = _build_bus(port)
    try:
        bus.connect()
    except Exception as exc:
        print(f"could not open serial bus: {exc}")
        return 1

    try:
        if not _verify_motors(bus):
            return 1
        neutral = _capture_neutral(bus)
        ranges = _capture_ranges(bus, neutral)
    finally:
        try:
            bus.enable_torque(False)
        except Exception:
            pass
        try:
            bus.disconnect()
        except Exception:
            pass

    payload = _build_calibration_payload(neutral, ranges)
    print()
    print("Calibration summary:")
    for name, mid in zip(MOTOR_NAMES, MOTOR_IDS, strict=True):
        n = payload["neutral_pos"][MOTOR_NAMES.index(name)]
        lo = payload["min_pos"][MOTOR_NAMES.index(name)]
        hi = payload["max_pos"][MOTOR_NAMES.index(name)]
        print(f"  motor {mid} {name:14s}: neutral={n:4d} min={lo:4d} max={hi:4d}")
    print()
    reply = input(f"Write to {CALIB_FILE}? [Y/n]: ").strip().lower()
    if reply in ("", "y", "yes"):
        _save(payload)
        print("done.")
        return 0
    print("aborted; nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
