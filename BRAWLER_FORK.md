# ExMR — brawler fork

This is a fork of [twehrbein/ExMR](https://github.com/twehrbein/ExMR), branch `brawler/up-from-camera`, based on release v0.1.5. It is used for non-commercial research under the original licence (see `LICENSE`). All copyright and attribution notices of ExMR and GVHMR are kept.

One fork is used everywhere — the ZeroGPU Space and local runs — so that every run has the same changes.

## Changes from v0.1.5

1. **`up_from_camera`** (off by default). It is available as `ExMR.forward(..., up_from_camera=True)`, `pipeline.Options.up_from_camera` and `tools/demo.py --up_from_camera`.
   - **The problem.** The network estimates the gravity direction separately in every frame, from the image alone. When the camera rolls or tilts during a shot, the estimate follows the rolled image, so the body leans by the camera's roll in the world frame. In one fight shot the camera rolled 28–30° and a standing person leaned 26–27°. The camera rotation (`cam_angvel`) was correct, but the rollout uses it only for yaw.
   - **The fix.** `global_traj.up_from_camera_gv` estimates the gravity direction once, over the first `UP_INIT_SECONDS` (0.5 s), in the camera of frame 0. It then carries that direction to every frame with the camera rotation, and rebuilds each frame's gravity-view frame from it before `rollout_global_params`. Everything after the rollout uses the corrected orientation: static-joint translation, `snap_to_ground`, IK and foot contacts.
   - **Diagnostics.** `forward` also returns `up_correction_deg` (B, L): the angle between the network's gravity direction and the carried one, per frame.
   - **Limits.** The result is only as good as the camera rotation from SimpleVO or DPVO. The first 0.5 s must show the true vertical; a wrong start keeps its error for the whole clip.
2. **SimpleVO guard.** A sampled frame with no SIFT features crashed FLANN with `cv2.error (-210) in buildIndex_`, for example a black frame at a cut. Now `_match` returns no matches for such a frame. A step with fewer than `MIN_MATCHES` (10) matches keeps the camera still, and the next frame is matched against the last good one. `SimpleVO.skipped_steps` counts these steps.

## Tests

The tests are CPU-only and need no weights. They are kept in the brawler lab run `2026-11-18-e12-exmr-space-up-01`:
- `test_fork_patch.py`, a synthetic scene: a 30° camera roll that the network takes for a lean drops from 29.5° to 0.00°; pure pitch and yaw, and a still camera, are unchanged.
- `test_vo_guard.py`: real SIFT on a black frame gives 0 matches and no error.
