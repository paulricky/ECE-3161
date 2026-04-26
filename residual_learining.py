from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

import values as val


@dataclass
class ResidualCorrection:
    delta_xyz: np.ndarray
    delta_rpy: np.ndarray
    enabled: bool
    source: str = "disabled"


class BoundedResidualCorrector:
    def __init__(self, model_path: Optional[str] = None):
        self.enabled = bool(getattr(val, "RESIDUAL_LEARNING_ENABLED", False))
        self.max_translation_m = float(getattr(val, "RESIDUAL_MAX_TRANSLATION_M", 0.02))
        self.max_rotation_rad = float(getattr(val, "RESIDUAL_MAX_ROTATION_RAD", 0.12))
        self.model_path = Path(model_path or getattr(val, "RESIDUAL_MODEL_FILE", "calibration_data/residual_model.json"))
        if not self.model_path.is_absolute():
            self.model_path = Path(__file__).resolve().parent / self.model_path
        self.bias_xyz = np.zeros(3, dtype=np.float64)
        self.bias_rpy = np.zeros(3, dtype=np.float64)
        self._load()

    def _load(self) -> None:
        if not self.enabled or not self.model_path.exists():
            return
        try:
            data = json.loads(self.model_path.read_text(encoding="utf-8"))
            self.bias_xyz = np.asarray(data.get("bias_xyz_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            self.bias_rpy = np.asarray(data.get("bias_rpy_rad", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        except Exception:
            self.enabled = False

    def predict(self, target_xyz, target_rpy=None, q_current=None, fk_xyz=None, vision_error=None) -> ResidualCorrection:
        if not self.enabled:
            return ResidualCorrection(np.zeros(3), np.zeros(3), False)
        dxyz = self.bias_xyz.copy()
        drpy = self.bias_rpy.copy()
        if vision_error is not None:
            dxyz += np.asarray(vision_error, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(dxyz))
        if norm > self.max_translation_m:
            dxyz *= self.max_translation_m / max(norm, 1e-12)
        rnorm = float(np.linalg.norm(drpy))
        if rnorm > self.max_rotation_rad:
            drpy *= self.max_rotation_rad / max(rnorm, 1e-12)
        return ResidualCorrection(dxyz, drpy, True, source="bounded_bias")

    def apply(self, target_xyz, target_rpy=None, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray], ResidualCorrection]:
        xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        rpy = None if target_rpy is None else np.asarray(target_rpy, dtype=np.float64).reshape(3)
        corr = self.predict(xyz, rpy, **kwargs)
        xyz2 = xyz + corr.delta_xyz
        rpy2 = None if rpy is None else rpy + corr.delta_rpy
        return xyz2, rpy2, corr