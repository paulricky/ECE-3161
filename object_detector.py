"""YOLO-based object detector for the pick-and-place pipeline.

Rate-limited: inference runs at most `inference_hz` times per second; the rest
of the time `maybe_detect` returns the cached result. Failure to load the
model (missing weights, no network) does not raise at import time — instead
the detector enters a disabled state and returns None detections.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    class_name: str
    confidence: float
    center_px: Tuple[float, float]
    pixel_angle_rad: float
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass
class DetectionFrame:
    timestamp: float
    frame_shape: Tuple[int, int]  # (H, W)
    detections: List[Detection] = field(default_factory=list)


def _wrap_symmetric_180(angle_rad: float) -> float:
    """Wrap an angle into [-pi/2, pi/2) — gripper is 180 degree symmetric."""
    a = float(angle_rad)
    a = math.fmod(a + math.pi / 2.0, math.pi)
    if a < 0:
        a += math.pi
    return a - math.pi / 2.0


def _long_axis_angle_from_bbox(
    frame: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    min_contour_area_ratio: float = 0.05,
) -> Optional[float]:
    """Estimate the object's long-axis angle from inside its bbox.

    Procedure: crop to bbox, Otsu-threshold the grayscale crop, pick the
    largest connected component, fit a minAreaRect, return the angle (in
    radians, in pixel-coordinate convention where +u is right and +v is
    down, wrapped to [-pi/2, pi/2)). Returns None if thresholding fails or
    no contour is big enough.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, min(x1, w_frame - 1))
    x2 = max(0, min(x2, w_frame))
    y1 = max(0, min(y1, h_frame - 1))
    y2 = max(0, min(y2, h_frame))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # Two variants so we survive both dark-object-on-light-table and inverse.
    _t, mask_a = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_b = cv2.bitwise_not(mask_a)

    min_area = max(25.0, min_contour_area_ratio * float(gray.shape[0] * gray.shape[1]))
    best_contour = None
    best_area = 0.0
    for mask in (mask_a, mask_b):
        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = float(cv2.contourArea(c))
            if area > best_area and area >= min_area:
                best_area = area
                best_contour = c

    if best_contour is None:
        return None

    (_cx, _cy), (w, h), angle_deg = cv2.minAreaRect(best_contour)

    # Normalise so angle describes the direction of the LONG axis.
    if w < h:
        angle_deg = angle_deg + 90.0

    return _wrap_symmetric_180(math.radians(float(angle_deg)))


class ObjectDetector:
    def __init__(
        self,
        model_path: str,
        class_whitelist: Optional[Sequence[str]] = None,
        conf_threshold: float = 0.45,
        device: str = "cpu",
        inference_hz: float = 8.0,
    ):
        self.model_path = str(model_path)
        self.class_whitelist = (
            set(str(c).lower() for c in class_whitelist) if class_whitelist else None
        )
        self.conf_threshold = float(conf_threshold)
        self.device = str(device)
        self.inference_period = 1.0 / max(1e-3, float(inference_hz))

        self.model = None
        self.class_names: dict = {}
        self.disabled: bool = False
        self._last_inference_time: float = 0.0
        self._last_result: Optional[DetectionFrame] = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # imported lazily so failures don't crash import
            self.model = YOLO(self.model_path)
            names = getattr(self.model, "names", None) or {}
            if isinstance(names, dict):
                self.class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, (list, tuple)):
                self.class_names = {int(i): str(v) for i, v in enumerate(names)}
            print(f"[object_detector] loaded model '{self.model_path}' "
                  f"with {len(self.class_names)} classes on device '{self.device}'")
        except Exception as exc:
            print(f"[object_detector] WARNING: failed to load YOLO model "
                  f"'{self.model_path}': {exc}")
            print("[object_detector] detector is DISABLED for this run.")
            self.disabled = True
            self.model = None

    def maybe_detect(self, frame: np.ndarray, now: Optional[float] = None) -> Optional[DetectionFrame]:
        if self.disabled:
            return None
        now = float(time.time() if now is None else now)
        if self._last_result is not None and (now - self._last_inference_time) < self.inference_period:
            return self._last_result

        result = self._run_inference(frame, now)
        self._last_result = result
        self._last_inference_time = now
        return result

    def _run_inference(self, frame: np.ndarray, now: float) -> DetectionFrame:
        detections: List[Detection] = []
        try:
            results = self.model(frame, verbose=False, device=self.device)
        except Exception as exc:
            print(f"[object_detector] inference error: {exc}")
            return DetectionFrame(timestamp=now, frame_shape=tuple(frame.shape[:2]),
                                  detections=[])

        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

            for i in range(len(boxes)):
                conf = float(confs[i])
                if conf < self.conf_threshold:
                    continue
                cls_id = int(clss[i])
                name = self.class_names.get(cls_id, str(cls_id))
                if self.class_whitelist is not None and name.lower() not in self.class_whitelist:
                    continue

                box = xyxy[i]
                x1 = int(round(float(box[0])))
                y1 = int(round(float(box[1])))
                x2 = int(round(float(box[2])))
                y2 = int(round(float(box[3])))
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                angle = _long_axis_angle_from_bbox(frame, (x1, y1, x2, y2))
                if angle is None:
                    angle = 0.0

                detections.append(Detection(
                    class_name=name,
                    confidence=conf,
                    center_px=(float(cx), float(cy)),
                    pixel_angle_rad=float(angle),
                    bbox_xyxy=(x1, y1, x2, y2),
                ))

        return DetectionFrame(
            timestamp=now,
            frame_shape=tuple(frame.shape[:2]),
            detections=detections,
        )

    @staticmethod
    def draw_overlay(frame: np.ndarray, detection: Detection,
                     color=(0, 200, 255), axis_length: int = 60) -> None:
        x1, y1, x2, y2 = detection.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        cx, cy = detection.center_px
        dx = math.cos(detection.pixel_angle_rad) * axis_length
        dy = math.sin(detection.pixel_angle_rad) * axis_length
        p0 = (int(round(cx - dx)), int(round(cy - dy)))
        p1 = (int(round(cx + dx)), int(round(cy + dy)))
        cv2.line(frame, p0, p1, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (int(round(cx)), int(round(cy))), 4, color, -1, cv2.LINE_AA)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _self_test() -> None:
    """Synthetic test of the orientation estimator (no YOLO required)."""
    canvas = np.full((400, 400, 3), 255, dtype=np.uint8)
    rect_center = (200, 200)
    rect_size = (160, 40)
    rot_deg = 30.0
    rot_rect = ((rect_center[0], rect_center[1]), rect_size, rot_deg)
    box = cv2.boxPoints(rot_rect).astype(np.int32)
    cv2.fillPoly(canvas, [box], (0, 0, 0))

    xs = box[:, 0]
    ys = box[:, 1]
    xyxy = (int(xs.min()) - 5, int(ys.min()) - 5, int(xs.max()) + 5, int(ys.max()) + 5)

    angle = _long_axis_angle_from_bbox(canvas, xyxy)
    assert angle is not None, "angle should not be None"
    # Expected: ~30 degrees, but because rectangle is horizontal at 0 rotation
    # the minAreaRect "first side" is the short side initially; after the
    # w<h swap we recover the long axis at 30 degrees.
    tol = math.radians(5.0)
    expected = _wrap_symmetric_180(math.radians(30.0))
    delta = abs(_wrap_symmetric_180(angle - expected))
    assert delta < tol, f"angle {math.degrees(angle):.2f} != {math.degrees(expected):.2f}"

    print("object_detector self-test OK")


if __name__ == "__main__":
    _self_test()
