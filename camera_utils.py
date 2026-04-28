from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2

import values as val


@dataclass
class CameraOpenResult:
    cap: cv2.VideoCapture
    frame: object
    index: int
    backend_name: str
    backend: Optional[int]
    used_props: bool


@dataclass
class CameraProbeResult:
    index: int
    backend_name: str
    opened: bool
    read_ok: bool
    frame_shape: Optional[tuple[int, ...]]
    used_props: bool
    message: str = ""


class CameraOpenError(RuntimeError):
    def __init__(self, message: str, attempts: list[CameraProbeResult]):
        super().__init__(message)
        self.attempts = attempts


def _unique_ints(items: Iterable[int]) -> list[int]:
    out: list[int] = []
    seen = set()
    for item in items:
        try:
            n = int(item)
        except Exception:
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def handtracking_candidate_indices() -> list[int]:
    primary = int(getattr(val, "HANDTRACKING_CAMERA_INDEX", 0))
    candidates = getattr(val, "HANDTRACKING_CAMERA_CANDIDATE_INDICES", [0, 1, 2])
    if not isinstance(candidates, (list, tuple)):
        candidates = [0, 1, 2]
    return _unique_ints([primary, *candidates])


def _backend_candidates() -> list[tuple[str, Optional[int]]]:
    configured = str(getattr(val, "HANDTRACKING_CAMERA_BACKEND", "auto")).strip().lower()
    avf = getattr(cv2, "CAP_AVFOUNDATION", None)
    is_macos = platform.system().lower() == "darwin"
    if configured == "avfoundation":
        return [("avfoundation", avf)] if avf is not None else [("default", None)]
    if configured == "default":
        return [("default", None)]
    if configured == "auto" and is_macos and avf is not None:
        return [("default", None), ("avfoundation", avf)]
    return [("default", None)]


def _open_capture(index: int, backend: Optional[int]) -> cv2.VideoCapture:
    if backend is None:
        return cv2.VideoCapture(int(index))
    return cv2.VideoCapture(int(index), int(backend))


def _apply_camera_props(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(getattr(val, "MAIN_CAMERA_BUFFERSIZE", 1)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(getattr(val, "MAIN_CAMERA_FRAME_WIDTH", 640)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(getattr(val, "MAIN_CAMERA_FRAME_HEIGHT", 480)))
    cap.set(cv2.CAP_PROP_FPS, int(getattr(val, "MAIN_CAMERA_FPS", 30)))


def _read_first_valid_frame(cap: cv2.VideoCapture):
    retries = max(1, int(getattr(val, "MAIN_CAMERA_READ_RETRIES", 30)))
    delay_s = max(0.0, float(getattr(val, "MAIN_CAMERA_READ_RETRY_DELAY_S", 0.05)))
    warmup = max(0, int(getattr(val, "MAIN_CAMERA_WARMUP_FRAMES", 5)))
    valid_count = 0
    first_valid = None

    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None:
            first_valid = frame
            valid_count += 1
            if valid_count > warmup:
                return frame
        if delay_s > 0.0:
            time.sleep(delay_s)
    return first_valid


def validate_live_camera_stream(
    cap: cv2.VideoCapture,
    required_frames: Optional[int] = None,
    timeout_s: Optional[float] = None,
):
    required = max(1, int(required_frames if required_frames is not None else getattr(val, "MAIN_CAMERA_STABILITY_FRAMES", 10)))
    timeout = max(0.1, float(timeout_s if timeout_s is not None else getattr(val, "MAIN_CAMERA_STABILITY_TIMEOUT_S", 3.0)))
    delay_s = max(0.0, float(getattr(val, "MAIN_CAMERA_READ_RETRY_DELAY_S", 0.05)))
    warmup = max(0, int(getattr(val, "MAIN_CAMERA_WARMUP_FRAMES", 5)))
    deadline = time.time() + timeout
    valid_frames = 0
    warmup_seen = 0
    last_frame = None

    while time.time() <= deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            if warmup_seen < warmup:
                warmup_seen += 1
            else:
                valid_frames += 1
                last_frame = frame
                if valid_frames >= required:
                    return last_frame
        elif delay_s > 0.0:
            time.sleep(delay_s)

    return last_frame if valid_frames >= required else None


def _try_camera(index: int, backend_name: str, backend: Optional[int], use_props: bool) -> tuple[Optional[CameraOpenResult], CameraProbeResult]:
    cap = _open_capture(index, backend)
    opened = bool(cap.isOpened())
    if opened and use_props:
        _apply_camera_props(cap)

    frame = validate_live_camera_stream(cap) if opened else None
    read_ok = frame is not None
    shape = tuple(frame.shape) if read_ok and hasattr(frame, "shape") else None
    probe = CameraProbeResult(
        index=int(index),
        backend_name=backend_name,
        opened=opened,
        read_ok=read_ok,
        frame_shape=shape,
        used_props=bool(use_props),
    )
    if read_ok:
        return CameraOpenResult(
            cap=cap,
            frame=frame,
            index=int(index),
            backend_name=backend_name,
            backend=backend,
            used_props=bool(use_props),
        ), probe
    try:
        cap.release()
    except Exception:
        pass
    return None, probe


def _print_probe(prefix: str, probe: CameraProbeResult) -> None:
    if not bool(getattr(val, "MAIN_CAMERA_VERBOSE_PROBE", True)):
        return
    shape = "" if probe.frame_shape is None else f" frame_shape={probe.frame_shape}"
    print(
        f"{prefix} index={probe.index} backend={probe.backend_name} "
        f"props={'yes' if probe.used_props else 'no'} opened={probe.opened} "
        f"read_ok={probe.read_ok}{shape}"
    )


def open_handtracking_camera() -> CameraOpenResult:
    attempts: list[CameraProbeResult] = []
    indices = handtracking_candidate_indices()
    backends = _backend_candidates()
    open_retries = max(1, int(getattr(val, "MAIN_CAMERA_OPEN_RETRIES", 3)))
    retry_without_props = bool(getattr(val, "MAIN_CAMERA_TRY_WITHOUT_PROP_SET_ON_FAIL", True))

    for index in indices:
        for backend_name, backend in backends:
            for _ in range(open_retries):
                result, probe = _try_camera(index, backend_name, backend, use_props=True)
                attempts.append(probe)
                _print_probe("[camera]", probe)
                if result is not None:
                    print(f"[camera] selected stable hand-tracking camera index={result.index} backend={result.backend_name} props=yes frame_shape={result.frame.shape}")
                    return result

            if retry_without_props:
                for _ in range(open_retries):
                    result, probe = _try_camera(index, backend_name, backend, use_props=False)
                    attempts.append(probe)
                    _print_probe("[camera]", probe)
                    if result is not None:
                        print(f"[camera] selected stable hand-tracking camera index={result.index} backend={result.backend_name} props=no frame_shape={result.frame.shape}")
                        return result

    fallback = open_previous_style_handtracking_camera(attempts)
    if fallback is not None:
        return fallback

    raise CameraOpenError("Could not read from hand-tracking camera.", attempts)


def open_specific_handtracking_camera(index: int, backend_name: str, used_props: bool) -> CameraOpenResult:
    backend = None
    for name, candidate in _backend_candidates():
        if name == backend_name:
            backend = candidate
            break
    result, probe = _try_camera(int(index), backend_name, backend, bool(used_props))
    _print_probe("[camera]", probe)
    if result is None:
        raise CameraOpenError("Could not reopen hand-tracking camera.", [probe])
    print(
        f"[camera] reopened stable hand-tracking camera index={result.index} "
        f"backend={result.backend_name} props={'yes' if result.used_props else 'no'} "
        f"frame_shape={result.frame.shape}"
    )
    return result


def open_previous_style_handtracking_camera(attempts: Optional[list[CameraProbeResult]] = None) -> Optional[CameraOpenResult]:
    print("[camera] using previous-style default OpenCV camera fallback")
    index = int(getattr(val, "HANDTRACKING_CAMERA_INDEX", 0))
    cap = cv2.VideoCapture(index)
    opened = bool(cap.isOpened())
    if opened:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(getattr(val, "MAIN_CAMERA_BUFFERSIZE", 1)))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(getattr(val, "MAIN_CAMERA_FRAME_WIDTH", 640)))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(getattr(val, "MAIN_CAMERA_FRAME_HEIGHT", 480)))
            cap.set(cv2.CAP_PROP_FPS, int(getattr(val, "MAIN_CAMERA_FPS", 30)))
        except Exception:
            pass
    frame = validate_live_camera_stream(cap) if opened else None
    probe = CameraProbeResult(
        index=index,
        backend_name="default",
        opened=opened,
        read_ok=frame is not None,
        frame_shape=tuple(frame.shape) if frame is not None and hasattr(frame, "shape") else None,
        used_props=True,
    )
    if attempts is not None:
        attempts.append(probe)
    _print_probe("[camera]", probe)
    if frame is not None:
        print(f"[camera] selected stable hand-tracking camera index={index} backend=default props=yes frame_shape={frame.shape}")
        return CameraOpenResult(cap=cap, frame=frame, index=index, backend_name="default", backend=None, used_props=True)
    try:
        cap.release()
    except Exception:
        pass
    return None


def print_camera_failure_help(exc: CameraOpenError) -> None:
    tried = sorted({p.index for p in exc.attempts})
    print("[main] Could not read from hand-tracking camera.")
    print(f"[main] Tried indices: {tried}")
    print("[main] Try changing HANDTRACKING_CAMERA_INDEX in values.py.")
    print("[main] On macOS, check System Settings > Privacy & Security > Camera and allow PyCharm/Terminal.")
    print("[main] Close other apps using the camera.")


def probe_camera_indices(indices: Iterable[int], read_retries: int = 8) -> list[CameraProbeResult]:
    old_retries = getattr(val, "MAIN_CAMERA_READ_RETRIES", 30)
    old_stability = getattr(val, "MAIN_CAMERA_STABILITY_FRAMES", 10)
    try:
        setattr(val, "MAIN_CAMERA_READ_RETRIES", int(read_retries))
        setattr(val, "MAIN_CAMERA_STABILITY_FRAMES", max(2, int(read_retries)))
        out: list[CameraProbeResult] = []
        for index in _unique_ints(indices):
            for backend_name, backend in _backend_candidates():
                result, probe = _try_camera(index, backend_name, backend, use_props=True)
                if result is not None:
                    result.cap.release()
                out.append(probe)
        return out
    finally:
        setattr(val, "MAIN_CAMERA_READ_RETRIES", old_retries)
        setattr(val, "MAIN_CAMERA_STABILITY_FRAMES", old_stability)
