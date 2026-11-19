# ExMR — brawler fork

This is a fork of [twehrbein/ExMR](https://github.com/twehrbein/ExMR), branch `brawler/up-from-camera`, based on release v0.1.5. It is used for non-commercial research under the original licence (see `LICENSE`). All copyright and attribution notices of ExMR and GVHMR are kept.

One fork is used everywhere — the ZeroGPU Space and local runs — so that every run has the same changes.

## Changes from v0.1.5

1. **`up_from_camera`** (off by default). It is available as `ExMR.forward(..., up_from_camera=True)`, `pipeline.Options.up_from_camera` and `tools/demo.py --up_from_camera`.
   - **The problem.** The network estimates the gravity direction separately in every frame, from the image alone. When the camera rolls or tilts during a shot, the estimate follows the rolled image, so the body leans by the camera's roll in the world frame. In one fight shot the camera rolled 28–30° and a standing person leaned 26–27°. The camera rotation (`cam_angvel`) was correct, but the rollout uses it only for yaw.
   - **The fix.** `global_traj.up_from_camera_gv` estimates the gravity direction once, over the first `UP_INIT_SECONDS` (0.5 s), in the camera of frame 0. It then carries that direction to every frame with the camera rotation, and rebuilds each frame's gravity-view frame from it before `rollout_global_params`. Everything after the rollout uses the corrected orientation: static-joint translation, `snap_to_ground`, IK and foot contacts.
   - **Diagnostics.** `forward` also returns `up_correction_deg` (B, L): the angle between the network's gravity direction and the carried one, per frame.
   - **Limits.** The result is only as good as the camera rotation from SimpleVO or DPVO. The first 0.5 s must show the true vertical; a wrong start keeps its error for the whole clip.
2. **SimpleVO guard.** A sampled frame with no SIFT features, for example a black frame at a cut, crashed FLANN with `cv2.error (-210) in buildIndex_`. Now a step that involves such a frame keeps the camera still, and the next frame is matched against the last frame that has features. `SimpleVO.skipped_steps` counts these steps. Frames that have features run exactly as upstream.
   - The first version (f4fe850) also skipped every step with fewer than 10 matches. That changed the camera on clips that upstream handles without a crash, so it was narrowed in v0.1.5-brawler.2.

3. **`vo_step`.** Sets SimpleVO's sampling step. It is available as `pipeline.Options.vo_step` (1–8, default 8 as upstream) and `tools/demo.py --vo_step`.
   - SimpleVO matches every `vo_step`-th frame and interpolates the rotation between them (SLERP). Smaller steps follow fast camera moves (whip pans, a hand-held camera) more closely, at more CPU time.
   - The step is part of the camera-rotation cache key.

4. **DPVO without the submodule** (branch `brawler/dpvo-probe`, not released yet). `DPVOOdometry` runs where exmr is installed with pip and `third_party/` is absent, as in the ZeroGPU Space.
   - DPVO's `config/default.yaml` of the pinned submodule (c0c5a10) is embedded as `DPVO_CONFIG`. Before, the wrapper read it from `third_party/DPVO/config/default.yaml`, which a pip install does not have.
   - `DPVO()` sets torch's CPU thread count to 2 for the whole process; the wrapper now restores the previous count when DPVO has finished.
   - DPVO itself (CUDA extensions, torch-scatter) is installed from wheels built for the Space's torch; see the brawler lab run `2026-11-19-e12-exmr-dpvo-probe-01`.

5. **`hard_static_hold`** (branch `brawler/static-hold-probe`, local probe, not released). Off by default. Available as `ExMR.forward(..., hard_static_hold=True)`, `pipeline.Options.hard_static_hold` and `tools/demo.py --hard_static_hold`.
   - Upstream's IK target of a static joint is `x[t] = c x[t-1] + (1 - c) fk[t]`: with a static confidence c of 0.8-0.9 the held foot still follows 10-20 % of the network's motion per frame, so it slides.
   - With the option, a confidence above `STATIC_CONF` (0.8, the threshold of `foot_contacts`) counts as 1: the joint is held exactly. The toe target stays within `TOE_TARGET_MAX_OFFSET` (3 cm) of the network's toe, as upstream.
   - The target recursion is now `global_traj.static_joint_targets`, testable without a body model.

6. **`side_static_hold`** (same probe branch). Off by default; implies `hard_static_hold`.
   - With `hard_static_hold` alone, a foot on the ground often keeps sliding: ExMR sets foot contact by the larger of the ankle and foot-joint confidences, and the ankle is frequently below 0.8, so only the foot joint is held, within 3 cm of the network's toe.
   - With the option, a foot whose larger confidence is above `STATIC_CONF` and whose ankle confidence is at least `SIDE_ANKLE_MIN` (0.5) is held as a whole (both joints), and the toe target may lie up to `SIDE_TOE_MAX_OFFSET` (0.10 m) from the network's toe.
   - The ankle threshold trades holding truly standing feet against sticking truly moving ones (Harmony4D truth: 81 % / 42 % of the contact frames at 0.5).

## Tests

The tests are CPU-only and need no weights. They are kept in the brawler lab run `2026-11-18-e12-exmr-space-up-01`:
- `test_fork_patch.py`, a synthetic scene: a 30° camera roll that the network takes for a lean drops from 29.5° to 0.00°; pure pitch and yaw, and a still camera, are unchanged.
- `test_vo_guard.py`: real SIFT. The black frame is detected as having no features; normal frames are not, and they still match (511 matches).
