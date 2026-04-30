from __future__ import annotations

import platform
import threading
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
    camera_mode: str = ""


@dataclass
class CameraProbeResult:
    index: int
    backend_name: str
    opened: bool
    read_ok: bool
    frame_shape: Optional[tuple[int, ...]]
    used_props: bool
    camera_mode: str = ""
    message: str = ""


class CameraOpenError(RuntimeError):
    def __init__(self, message: str, attempts: list[CameraProbeResult]):
        super().__init__(message)
        self.attempts = attempts


@dataclass
class LatestFrame:
    ok: bool
    frame: object
    timestamp: float
    age_s: float
    sequence: int
    failures: int


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


def _camera_resolution_for_log() -> str:
    width = int(getattr(val, "CAMERA_CAPTURE_WIDTH", getattr(val, "MAIN_CAMERA_FRAME_WIDTH", 640)))
    height = int(getattr(val, "CAMERA_CAPTURE_HEIGHT", getattr(val, "MAIN_CAMERA_FRAME_HEIGHT", 480)))
    fps = int(getattr(val, "CAMERA_CAPTURE_FPS", getattr(val, "MAIN_CAMERA_FPS", 30)))
    return f"{width}x{height}@{fps}"


def _use_native_handtracking_view() -> bool:
    return (
        bool(getattr(val, "HANDTRACKING_CAMERA_USE_NATIVE_VIEW", False))
        or not bool(getattr(val, "HANDTRACKING_CAMERA_FORCE_RESOLUTION", True))
    )


def _camera_prop_mode_label(use_props: bool) -> str:
    if not use_props:
        return "unconfigured"
    if _use_native_handtracking_view():
        return "native_view"
    return f"forced_resolution:{_camera_resolution_for_log()}"


def _apply_camera_props(cap: cv2.VideoCapture) -> None:
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(getattr(val, "MAIN_CAMERA_BUFFERSIZE", 1)))
        except Exception:
            pass
    if bool(getattr(val, "CAMERA_FORCE_MJPEG", False)):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if _use_native_handtracking_view():
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(getattr(val, "CAMERA_CAPTURE_WIDTH", getattr(val, "MAIN_CAMERA_FRAME_WIDTH", 640))))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(getattr(val, "CAMERA_CAPTURE_HEIGHT", getattr(val, "MAIN_CAMERA_FRAME_HEIGHT", 480))))
    cap.set(cv2.CAP_PROP_FPS, int(getattr(val, "CAMERA_CAPTURE_FPS", getattr(val, "MAIN_CAMERA_FPS", 30))))


def read_latest_from_capture(cap: cv2.VideoCapture):
    if bool(getattr(val, "CAMERA_FLUSH_STALE_FRAMES", True)):
        flush_count = max(0, int(getattr(val, "CAMERA_FLUSH_COUNT", 3)))
        for _ in range(flush_count):
            try:
                if not cap.grab():
                    break
            except Exception:
                break
        try:
            return cap.retrieve()
        except Exception:
            return False, None
    return cap.read()


class LatestFrameCamera:
    def __init__(self, cap: cv2.VideoCapture, initial_frame=None, initial_timestamp: Optional[float] = None):
        self.cap = cap
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame = initial_frame
        self._timestamp = float(initial_timestamp if initial_timestamp is not None else time.time())
        self._sequence = 1 if initial_frame is not None else 0
        self._failures = 0
        self._reads = 0
        self._start_time = time.time()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="latest-frame-camera", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        delay_s = max(0.001, float(getattr(val, "MAIN_CAMERA_READ_RETRY_DELAY_S", 0.05)))
        while not self._stop.is_set():
            ok, frame = read_latest_from_capture(self.cap)
            now = time.time()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._timestamp = now
                    self._sequence += 1
                    self._failures = 0
                    self._reads += 1
            else:
                with self._lock:
                    self._failures += 1
                self._stop.wait(delay_s)

    def read_latest(self) -> LatestFrame:
        now = time.time()
        with self._lock:
            frame = self._frame
            ts = self._timestamp
            seq = self._sequence
            failures = self._failures
        return LatestFrame(
            ok=frame is not None,
            frame=frame,
            timestamp=ts,
            age_s=max(0.0, now - ts),
            sequence=seq,
            failures=failures,
        )

    def stats(self) -> dict:
        elapsed = max(time.time() - self._start_time, 1e-6)
        with self._lock:
            reads = self._reads
            failures = self._failures
        return {"capture_fps": reads / elapsed, "queue_size": 1 if self._frame is not None else 0, "failures": failures}

    def release(self) -> None:
        self._stop.set()
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


def _read_first_valid_frame(cap: cv2.VideoCapture):
    retries = max(1, int(getattr(val, "MAIN_CAMERA_READ_RETRIES", 30)))
    delay_s = max(0.0, float(getattr(val, "MAIN_CAMERA_READ_RETRY_DELAY_S", 0.05)))
    warmup = max(0, int(getattr(val, "MAIN_CAMERA_WARMUP_FRAMES", 5)))
    valid_count = 0
    first_valid = None

    for _ in range(retries):
        ok, frame = read_latest_from_capture(cap)
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
    camera_mode = _camera_prop_mode_label(use_props)

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
        camera_mode=camera_mode,
    )
    if read_ok:
        return CameraOpenResult(
            cap=cap,
            frame=frame,
            index=int(index),
            backend_name=backend_name,
            backend=backend,
            used_props=bool(use_props),
            camera_mode=camera_mode,
        ), probe
    try:
        cap.release()
    except Exception:
        pass
    return None, probe


def _print_probe(prefix: str, probe: CameraProbeResult) -> None:
    if not bool(getattr(val, "MAIN_CAMERA_VERBOSE_PROBE", True)):
        return
    print_shape = bool(getattr(val, "HANDTRACKING_CAMERA_PRINT_FRAME_SHAPE", True))
    shape = "" if probe.frame_shape is None or not print_shape else f" frame_shape={probe.frame_shape}"
    print(
        f"{prefix} index={probe.index} backend={probe.backend_name} "
        f"props={'yes' if probe.used_props else 'no'} opened={probe.opened} "
        f"read_ok={probe.read_ok} mode={probe.camera_mode}{shape}"
    )


def _selected_camera_message(prefix: str, result: CameraOpenResult) -> str:
    shape = ""
    if bool(getattr(val, "HANDTRACKING_CAMERA_PRINT_FRAME_SHAPE", True)) and hasattr(result.frame, "shape"):
        shape = f" frame_shape={result.frame.shape}"
    return (
        f"{prefix} stable hand-tracking camera index={result.index} backend={result.backend_name} "
        f"props={'yes' if result.used_props else 'no'} mode={result.camera_mode}{shape}"
    )


def open_handtracking_camera(candidate_indices: Optional[Iterable[int]] = None) -> CameraOpenResult:
    attempts: list[CameraProbeResult] = []
    indices = handtracking_candidate_indices() if candidate_indices is None else _unique_ints(candidate_indices)
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
                    print(_selected_camera_message("[camera] selected", result))
                    return result

            if retry_without_props:
                for _ in range(open_retries):
                    result, probe = _try_camera(index, backend_name, backend, use_props=False)
                    attempts.append(probe)
                    _print_probe("[camera]", probe)
                    if result is not None:
                        print(_selected_camera_message("[camera] selected", result))
                        return result

    fallback = open_previous_style_handtracking_camera(attempts, index=indices[0] if indices else None)
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
        _selected_camera_message("[camera] reopened", result)
    )
    return result


def open_previous_style_handtracking_camera(attempts: Optional[list[CameraProbeResult]] = None, index: Optional[int] = None) -> Optional[CameraOpenResult]:
    print("[camera] using previous-style default OpenCV camera fallback")
    index = int(getattr(val, "HANDTRACKING_CAMERA_INDEX", 0) if index is None else index)
    cap = cv2.VideoCapture(index)
    opened = bool(cap.isOpened())
    if opened:
        try:
            _apply_camera_props(cap)
        except Exception:
            pass
    frame = validate_live_camera_stream(cap) if opened else None
    camera_mode = _camera_prop_mode_label(True)
    probe = CameraProbeResult(
        index=index,
        backend_name="default",
        opened=opened,
        read_ok=frame is not None,
        frame_shape=tuple(frame.shape) if frame is not None and hasattr(frame, "shape") else None,
        used_props=True,
        camera_mode=camera_mode,
    )
    if attempts is not None:
        attempts.append(probe)
    _print_probe("[camera]", probe)
    if frame is not None:
        result = CameraOpenResult(cap=cap, frame=frame, index=index, backend_name="default", backend=None, used_props=True, camera_mode=camera_mode)
        print(_selected_camera_message("[camera] selected", result))
        return result
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
    retry_without_props = bool(getattr(val, "MAIN_CAMERA_TRY_WITHOUT_PROP_SET_ON_FAIL", True))
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
                if probe.read_ok or not retry_without_props:
                    continue
                result, probe = _try_camera(index, backend_name, backend, use_props=False)
                if result is not None:
                    result.cap.release()
                out.append(probe)
        return out
    finally:
        setattr(val, "MAIN_CAMERA_READ_RETRIES", old_retries)
        setattr(val, "MAIN_CAMERA_STABILITY_FRAMES", old_stability)
