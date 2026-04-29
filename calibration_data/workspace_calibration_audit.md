# Workspace Calibration Audit

Verdict: **BAD**

## Critical
- None

## Warnings
- left/right are not mostly opposite (cos=0.36); redo one or both poses.
- down/up are not mostly opposite (cos=0.85); redo one or both poses.
- far/near are not mostly opposite (cos=-0.03); redo one or both poses.
- Pose left has strong off-axis coupling; redo if hand mapping feels skewed.
- Pose near has strong off-axis coupling; redo if hand mapping feels skewed.
- Optional pose up_left residual 0.395 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose up_right residual 0.335 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose down_left residual 0.248 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose down_right residual 0.232 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose near_left residual 0.124 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose near_right residual 0.122 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose far_left residual 0.197 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose far_right residual 0.228 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose near_up residual 0.224 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose near_down residual 0.418 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose far_up residual 0.340 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose far_down residual 0.558 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose near_up_left residual 0.380 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose near_up_right residual 0.388 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose far_down_left residual 0.529 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Optional pose far_down_right residual 0.547 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- Legacy HAND_TARGET/WORKSPACE bounds would clip 48 recorded pose coordinates; runtime should use calibrated workspace bounds.

## Recommended Redo Poses
- down_left: Optional pose down_left residual 0.248 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- down_right: Optional pose down_right residual 0.232 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- far_down: Optional pose far_down residual 0.558 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- far_down_left: Optional pose far_down_left residual 0.529 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- far_down_right: Optional pose far_down_right residual 0.547 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- far_left: Optional pose far_left residual 0.197 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- far_right: Optional pose far_right residual 0.228 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- far_up: Optional pose far_up residual 0.340 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- left: Pose left has strong off-axis coupling; redo if hand mapping feels skewed.
- near: Pose near has strong off-axis coupling; redo if hand mapping feels skewed.
- near_down: Optional pose near_down residual 0.418 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- near_left: Optional pose near_left residual 0.124 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- near_right: Optional pose near_right residual 0.122 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- near_up: Optional pose near_up residual 0.224 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- near_up_left: Optional pose near_up_left residual 0.380 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- near_up_right: Optional pose near_up_right residual 0.388 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- up_left: Optional pose up_left residual 0.395 m exceeds clamp by more than 2x; redo it or fix base extrema first.
- up_right: Optional pose up_right residual 0.335 m exceeds clamp by more than 2x; redo it or fix base extrema first.
