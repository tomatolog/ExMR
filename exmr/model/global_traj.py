"""World-frame trajectory: gravity-view rollout and static-joint post-processing (from GVHMR),
putting the motion on the ground, optionally snapped to a flat ground (as in PRIMAL), facing +z,
and the feet's contacts with the ground."""

import torch
import torch.nn.functional as F

from exmr.geometry import kinematics
from exmr.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from exmr.geometry.smoothing import gaussian_smooth, moving_average_smooth
from exmr.model.ik import CCDIK

# joints with a static (not moving) prediction: [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
STATIC_JOINT_IDS = [7, 10, 8, 11, 20, 21]
# snap_to_ground: some body part touches the ground at least this often (seconds), about the hang
# time of a jump (as in PRIMAL, Zhang et al., arXiv 2503.17544)
CONTACT_INTERVAL = 0.5
# static_joint_ik: a toe's IK target stays this close (m, per axis) to the network's toe
TOE_TARGET_MAX_OFFSET = 0.03
# (brawler fork) side_static_hold: a side is held as a whole when its ankle's static confidence is at least
# SIDE_ANKLE_MIN; its toe target may then lie up to SIDE_TOE_MAX_OFFSET (m) from the network's toe
SIDE_ANKLE_MIN = 0.5
SIDE_TOE_MAX_OFFSET = 0.10
# foot_contacts: a foot touches the ground where its ankle or toe is predicted static (not
# moving) with at least STATIC_CONF and it is at most FOOT_CONTACT_HEIGHT (m) above the lowest
# point of the body
STATIC_CONF = 0.8
FOOT_CONTACT_HEIGHT = 0.05
# static_cam_transl: the world root may deviate this far (m) from the camera-frame root before
# the slow part of the deviation (Gaussian smoothing, sigma in seconds) is removed
CAMERA_DEAD_BAND = 0.25
CAMERA_DRIFT_SIGMA = 2.0
# up_from_camera: the gravity direction is estimated over this many seconds at the start
UP_INIT_SECONDS = 0.5
# (brawler fork) unseen feet: a foot is seen where any of its COCO-23 keypoints (ankle, big toe,
# small toe, heel) has a confidence above FOOT_SEEN_CONF, the body network's own visibility
# threshold (body_net.py); where it is not seen the network gets no evidence of it and predicts
# no static confidence, so its contact comes from a rule on the body (UNSEEN_FOOT_RULES) instead
FOOT_KEYPOINTS = ((15, 17, 18, 19), (16, 20, 21, 22))  # left, right
FOOT_SEEN_CONF = 0.5
UNSEEN_FOOT_RULES = ("none", "lower", "standing")
# a foot given contact by the rule gets this static logit (sigmoid 0.9975, above STATIC_CONF)
UNSEEN_HOLD_LOGIT = 6.0
# rule "lower": the lower foot stands (someone always stands unless lying); both feet stand where
# their lowest joints are within LOWER_BOTH (m); a stance shorter than LOWER_MIN_SECONDS joins its
# neighbour; no stance where the pelvis is less than LYING_PELVIS (m) above the lower ankle
LOWER_BOTH = 0.02
LOWER_MIN_SECONDS = 0.1
LYING_PELVIS = 0.40
# rule "standing" (run 2026-11-17-e12-standing-rule-01, feature v2): both feet stand where, within
# +- STANDING_WINDOW seconds, the ankles keep their distance (STANDING_DIST range, m), direction
# (STANDING_ANGLE range, deg, where at least STANDING_ANGLE_MIN_DIST apart) and distance from the
# hip centre (STANDING_RADIUS range, m), the hip line tilts at most STANDING_TILT (deg), for at
# least STANDING_MIN_SECONDS; ankles smoothed with a Gaussian of STANDING_SIGMA seconds
STANDING_WINDOW, STANDING_SIGMA, STANDING_MIN_SECONDS = 0.25, 0.042, 0.25
STANDING_DIST, STANDING_ANGLE, STANDING_ANGLE_MIN_DIST = 0.12, 30.0, 0.10
STANDING_RADIUS, STANDING_TILT = 0.25, 30.0
# hold_source codes (per frame and foot)
HOLD_NONE, HOLD_NETWORK, HOLD_UNSEEN_RULE = 0, 1, 2


def up_from_camera_gv(global_orient_gv, global_orient_c, cam_angvel, init_frames):
    """Gravity-view root orientations whose gravity follows the camera rotation (brawler fork).

    The network estimates the gravity direction in every frame from that frame alone; when the
    camera rolls or tilts during the video (a hand-held or rolling shot), the estimate follows the
    rolled image and the body leans by the camera's roll in the world frame. Here the gravity
    direction is estimated once, from the first init_frames frames, in the camera of frame 0, and
    carried to every frame by the camera rotation (cam_angvel), so the camera's roll and tilt are
    not taken for the body's. The camera-frame root orientation (global_orient_c) is kept; the
    GV frame of each frame is rebuilt from the carried gravity and the camera's view direction.

    Args:
        global_orient_gv: (B, L, 3) root orientation in the per-frame GV frame (network)
        global_orient_c: (B, L, 3) root orientation in the camera frame
        cam_angvel: (B, L, 6) camera rotation from frame t to t+1 (6D), R @ R_w2c^t = R_w2c^t+1
        init_frames: frames at the start over which the gravity direction is averaged
    Returns:
        (B, L, 3) root orientation in the rebuilt GV frames, and (B, L) angle (degrees) between the
        network's and the carried gravity direction per frame (the correction applied)
    """
    B, L = global_orient_gv.shape[:2]
    R_step = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3), frame t -> t+1
    R_w2c = [torch.eye(3, dtype=R_step.dtype, device=R_step.device).expand(B, 3, 3)]
    for t in range(L - 1):
        R_w2c.append(R_step[:, t] @ R_w2c[-1])
    R_w2c = torch.stack(R_w2c, dim=1)  # (B, L, 3, 3): camera of frame 0 -> camera of frame t
    R_gv = axis_angle_to_matrix(global_orient_gv)
    R_c = axis_angle_to_matrix(global_orient_c)
    R_c2gv = R_gv @ R_c.mT  # camera -> GV (network), per frame
    down_c = R_c2gv[..., 1, :]  # GV y (down) in camera coordinates, per frame (network)
    down_0 = (R_w2c.mT @ down_c[..., None])[..., 0]  # ... in the camera of frame 0
    k = max(1, min(int(init_frames), L))
    d0 = F.normalize(down_0[:, :k].mean(dim=1), dim=-1)  # (B, 3)
    down_t = F.normalize((R_w2c @ d0[:, None, :, None])[..., 0], dim=-1)  # carried, camera of t
    view = torch.tensor([0.0, 0.0, 1.0], dtype=R_c.dtype, device=R_c.device).expand_as(down_t)
    z = F.normalize(view - (view * down_t).sum(-1, keepdim=True) * down_t, dim=-1)
    x = torch.linalg.cross(down_t, z, dim=-1)
    R_c2gv_new = torch.stack([x, down_t, z], dim=-2)  # rows: GV axes in camera coordinates
    correction = torch.rad2deg(
        torch.acos(torch.clamp((F.normalize(down_c, dim=-1) * down_t).sum(-1), -1.0, 1.0))
    )
    return matrix_to_axis_angle(R_c2gv_new @ R_c), correction


def rollout_global_params(global_orient_gv, local_transl_vel, global_orient_c, cam_angvel):
    """World-frame root orientation and translation, starting at the origin in frame 0.

    The gravity-view (GV) frame of every frame is aligned to the first one by the camera's yaw
    rotation (from the camera angular velocity), then the root velocity is integrated.

    Args:
        global_orient_gv: (B, L, 3) root orientation in the per-frame GV frame
        local_transl_vel: (B, L, 3) root velocity in the root's frame
        global_orient_c: (B, L, 3) root orientation in the camera frame
        cam_angvel: (B, L, 6) camera rotation from frame t to t+1 (6D), R @ R_w2c^t = R_w2c^t+1
    Returns:
        dict with global_orient (B, L, 3) and transl (B, L, 3), y-axis up
    """

    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R

    B = cam_angvel.shape[0]
    R_t_to_tp1 = as_identity(rotation_6d_to_matrix(cam_angvel))  # (B, L, 3, 3)

    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (B, L, 3, 3)

    # camera view direction in the GV frame (its x-axis is estimated, so != 0)
    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[:, :, :, 2]  # (B, L, 3)
    # ... and in the next frame
    view_axis_gv_next = (R_c2gv @ R_t_to_tp1.mT)[..., 2]

    # yaw between consecutive GV frames: angle between the view directions projected to xz
    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = F.normalize(vec1_xyz, dim=-1)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = F.normalize(vec2_xyz, dim=-1)

    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)
    aa_tp1_to_t_angle = torch.acos(
        torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0)
    )
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1) * aa_tp1_to_t_angle
    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT  # (B, L, 3, 3)

    # accumulate: GV frame of t -> GV frame of 0
    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = as_identity(torch.stack(R_t_to_0, dim=1))  # (B, L, 3, 3)

    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)

    # integrate the root velocity (in the root frame) starting at the origin
    transl_vel = torch.einsum(
        "...lij,...lj->...li", axis_angle_to_matrix(global_orient), local_transl_vel
    )
    transl = torch.cumsum(
        torch.cat([torch.zeros_like(transl_vel[..., :1, :]), transl_vel[..., :-1, :]], dim=-2),
        dim=-2,
    )

    # GV is y-down: rotate by 180 degrees around z to make y point up
    R_flip = axis_angle_to_matrix(torch.tensor([0, 0, torch.pi]).to(global_orient.device))
    global_orient = matrix_to_axis_angle(R_flip @ axis_angle_to_matrix(global_orient))
    transl = torch.einsum("...ij,...j->...i", R_flip, transl)
    return {"global_orient": global_orient, "transl": transl}


def static_joint_transl(smplx_global, static_conf_logits, decoder):
    """Moving camera: correct the world translation so that joints predicted static stay put.

    Returns the post-processed (B, L, 3) translation.
    """
    pred_w_j3d = decoder.fk(**smplx_global)  # (B, L, 22, 3)
    pred_j3d_static = pred_w_j3d.clone()[:, :, STATIC_JOINT_IDS]  # (B, L, J, 3)
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)

    static_conf_logits = static_conf_logits[:, :-1].clone()
    static_label = static_conf_logits > 0  # (B, L-1, J)
    # non-static joints get a large negative logit, i.e. zero weight after the softmax
    static_conf_logits = static_conf_logits.float() - (~static_label * 1e6)
    is_static = static_label.sum(dim=-1) > 0  # (B, L-1), at least one joint static
    # per-frame displacement of the static joints, weighted by their confidence
    pred_disp = pred_j_disp * static_conf_logits[..., None].softmax(dim=-2)  # (B, L-1, J, 3)
    pred_disp = (pred_disp * is_static[..., None, None]).sum(-2)  # (B, L-1, 3)

    # subtract that displacement from the root translation, re-integrate, smooth x and z
    pred_w_transl = smplx_global["transl"].clone()  # (B, L, 3)
    pred_w_disp_new = (pred_w_transl[:, 1:] - pred_w_transl[:, :-1]) - pred_disp
    post_w_transl = torch.cumsum(torch.cat([pred_w_transl[:, :1], pred_w_disp_new], dim=1), dim=1)
    post_w_transl[..., 0] = gaussian_smooth(post_w_transl[..., 0], dim=-1)
    post_w_transl[..., 2] = gaussian_smooth(post_w_transl[..., 2], dim=-1)
    return post_w_transl


def static_cam_transl(smplx_incam, smplx_global, static_conf_logits, decoder, fps):
    """Static camera: the moving-camera correction (static_joint_transl), then the slow drift of
    the world root away from the camera-frame root, which a static camera places in the world, is
    removed where it exceeds CAMERA_DEAD_BAND. Returns the post-processed (B, L, 3) translation.
    """
    transl = static_joint_transl(smplx_global, static_conf_logits, decoder)
    world_root = decoder.fk(**{**smplx_global, "transl": transl})[:, :, 0]  # (B, L, 3)
    # the camera-frame root, smoothed as it could be noisy
    smplx_incam = {**smplx_incam, "transl": gaussian_smooth(smplx_incam["transl"], 5, dim=-2)}
    cam_root = decoder.fk(**smplx_incam)[:, :, 0]  # (B, L, 3)

    # into the world with the camera-to-world rotation of the first frame, where both roots meet
    R_c2w = (
        axis_angle_to_matrix(smplx_global["global_orient"][:, 0])
        @ axis_angle_to_matrix(smplx_incam["global_orient"][:, 0]).mT
    )  # (B, 3, 3)
    cam_root_in_w = torch.einsum("bij,blj->bli", R_c2w, cam_root - cam_root[:, :1])
    cam_root_in_w = cam_root_in_w + world_root[:, :1]

    # the slow part of the deviation, beyond the dead band (per axis)
    drift = gaussian_smooth(world_root - cam_root_in_w, CAMERA_DRIFT_SIGMA * fps, dim=-2)
    drift = drift.sign() * (drift.abs() - CAMERA_DEAD_BAND).clamp(min=0)
    return transl - drift


def lowest_vertex_height(smplx_global, body_model, vertex_sets=None, chunk_size=256):
    """Height (y) of the lowest SMPL-X mesh vertex in every frame, (B, L); with vertex_sets (a
    list of S vertex id tensors, e.g. the feet, or slice(None) for all vertices), of the lowest
    vertex of each set, (B, L, S).

    The mesh is computed in chunks of frames: all of a long video's would take gigabytes.
    """

    def lowest(height):  # (N, V) -> (N) or (N, S)
        if vertex_sets is None:
            return height.amin(dim=-1)
        return torch.stack([height[:, ids].amin(dim=-1) for ids in vertex_sets], dim=-1)

    params = {k: v.flatten(0, 1) for k, v in smplx_global.items()}  # (B * L, D)
    heights = [
        lowest(body_model(**{k: v[i : i + chunk_size] for k, v in params.items()})[..., 1])
        for i in range(0, len(params["transl"]), chunk_size)
    ]
    heights = torch.cat(heights)
    return heights.view(*smplx_global["transl"].shape[:2], *heights.shape[1:])


def put_on_ground(smplx_global, body_model):
    """Move the motion vertically so that the lowest mesh vertex over all frames is on the
    ground (y=0). Returns the (B, L, 3) translation."""
    ground = lowest_vertex_height(smplx_global, body_model).amin(dim=-1)  # (B,)
    transl = smplx_global["transl"].clone()
    transl[..., 1] -= ground[:, None]
    return transl


def snap_to_ground(smplx_global, body_model, fps, interval=CONTACT_INTERVAL):
    """Flat ground and nothing to stand on: move the body vertically so that it touches the
    ground (y=0) at least every `interval` seconds, which removes vertical drift.

    The ground height at a frame is the lowest mesh vertex within the window of about `interval`
    seconds around it (a sliding minimum), smoothed over the same window. Vertical motion within
    the window is kept, e.g. jumps with a shorter flight; longer flights are pulled down.
    Returns the (B, L, 3) translation.
    """
    lowest = lowest_vertex_height(smplx_global, body_model)  # (B, L)
    L = lowest.shape[1]
    k = min(2 * int(interval * fps / 2) + 1, L - 1 + L % 2)  # odd window size, at most L
    ground = -F.max_pool1d(-lowest[:, None], k, stride=1)  # (B, 1, L - k + 1)
    # at the sequence ends, the windows are shifted inward instead of cut off
    ground = F.pad(ground, (k // 2, k // 2), mode="replicate")[:, 0]  # (B, L)
    ground = moving_average_smooth(ground, window_size=k, dim=-1)
    transl = smplx_global["transl"].clone()
    transl[..., 1] -= ground
    return transl


def face_forward(smplx_global, decoder):
    """Rotate the motion about the vertical axis through the first frame's root so that the body
    faces +z in the first frame (from its hips and shoulders), and move that root to x=z=0.
    Returns the (B, L, 3) global orientation and translation."""
    joints = decoder.fk(**{k: v[:, :1] for k, v in smplx_global.items()})[:, 0]  # (B, 22, 3)
    left = joints[:, 1] - joints[:, 2] + joints[:, 16] - joints[:, 17]  # (B, 3) hips, shoulders
    # the rotation about y that turns the left direction to +x (then the body faces +z)
    yaw = torch.atan2(left[:, 2], left[:, 0])
    yaw = torch.where(left[:, [0, 2]].norm(dim=-1) > 1e-4, yaw, torch.zeros_like(yaw))
    zeros = torch.zeros_like(yaw)
    R = axis_angle_to_matrix(torch.stack([zeros, yaw, zeros], dim=-1))  # (B, 3, 3)
    global_orient = matrix_to_axis_angle(
        R[:, None] @ axis_angle_to_matrix(smplx_global["global_orient"])
    )
    # SMPL-X rotates the body about its rest-pose pelvis J0: the root is at J0 + transl
    J0 = decoder.skeleton(smplx_global["betas"])[..., 0, :]  # (B, L, 3)
    root = smplx_global["transl"] + J0
    start = root[:, :1] * root.new_tensor([1.0, 0.0, 1.0])  # the first root on the ground plane
    transl = torch.einsum("bij,blj->bli", R, root - start) - J0
    return global_orient, transl


def foot_contacts(smplx_global, static_conf_logits, body_model):
    """Where the feet touch the ground, (B, L, 2) bool [left, right], and the height of each
    foot's lowest mesh vertex, (B, L, 2).

    A foot touches the ground where its ankle or toe is predicted static (STATIC_CONF) and it is
    the lowest part of the body (within FOOT_CONTACT_HEIGHT): not where a foot is held still in
    the air, e.g. lying on the back with the legs up.
    """
    feet = [body_model.left_foot_vertices, body_model.right_foot_vertices]
    heights = lowest_vertex_height(smplx_global, body_model, [*feet, slice(None)])  # (B, L, 3)
    conf = static_conf_logits.sigmoid()  # (B, L, 6): [L_ankle, L_foot, R_ankle, R_foot, ...]
    static = torch.stack([conf[..., :2].amax(dim=-1), conf[..., 2:4].amax(dim=-1)], dim=-1)
    lowest = heights[..., 2:] + FOOT_CONTACT_HEIGHT
    return (static > STATIC_CONF) & (heights[..., :2] <= lowest), heights[..., :2]


def feet_seen(kp2d):
    """(brawler fork) (B, L, 2) bool [left, right]: the foot has a keypoint above FOOT_SEEN_CONF.

    kp2d: (B, L, J >= 23, 3) COCO keypoints in pixels [x, y, conf]
    """
    return torch.stack(
        [(kp2d[..., list(ids), 2] > FOOT_SEEN_CONF).any(dim=-1) for ids in FOOT_KEYPOINTS], dim=-1
    )


def _merge_short_runs(labels, min_len):
    """1D integer labels: a run shorter than min_len takes its previous (else next) run's label."""
    labels = labels.clone()
    runs, start = [], 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[start]:
            runs.append([start, t])
            start = t
    for i, (a, b) in enumerate(runs):
        if b - a < min_len and len(runs) > 1:
            labels[a:b] = labels[runs[i - 1][0]] if i > 0 else labels[runs[i + 1][0]]
    return labels


def unseen_stance(joints, fps, rule):
    """(brawler fork) (B, L, 2) bool [left, right]: where the rule gives the foot contact.

    joints: (B, L, 22, 3) world-frame body joints (y up) of the network's motion, before the
    post-processing (only their relative geometry is used, not the root's drifting path).
    rule: "lower" or "standing" (see UNSEEN_FOOT_RULES and the constants).
    """
    ankles, foot_joints = joints[..., [7, 8], :], joints[..., [10, 11], :]
    lowest = torch.minimum(ankles[..., 1], foot_joints[..., 1])  # (B, L, 2)
    lying = joints[..., 0, 1] - lowest.min(dim=-1).values < LYING_PELVIS  # (B, L)
    if rule == "lower":
        d = lowest[..., 0] - lowest[..., 1]  # > 0: the left foot is higher
        labels = torch.where(d.abs() <= LOWER_BOTH, 2, torch.where(d > 0, 1, 0))
        min_len = max(1, round(LOWER_MIN_SECONDS * fps))
        labels = torch.stack([_merge_short_runs(lab, min_len) for lab in labels])
        stance = torch.stack([labels != 1, labels != 0], dim=-1)  # 0 left, 1 right, 2 both
    elif rule == "standing":
        sig = STANDING_SIGMA * fps
        pts = joints[..., [1, 2, 7, 8], :]  # hips, ankles
        pts = gaussian_smooth(pts.transpose(-3, -1), sig, dim=-1).transpose(-3, -1) if sig > 0.3 else pts
        hl, hr, al, ar = pts.unbind(dim=-2)
        centre = (hl + hr) / 2
        lat = hl - hr
        tilt = torch.rad2deg(torch.asin((lat[..., 1].abs() / lat.norm(dim=-1).clamp_min(1e-9)).clamp(0, 1)))
        dv = (al - ar)[..., [0, 2]]
        dist = dv.norm(dim=-1)
        theta = torch.rad2deg(torch.atan2(dv[..., 1], dv[..., 0]))
        r_l = (al - centre)[..., [0, 2]].norm(dim=-1)
        r_r = (ar - centre)[..., [0, 2]].norm(dim=-1)
        w = round(STANDING_WINDOW * fps)
        B, L = dist.shape
        rows = torch.zeros(B, L, dtype=torch.bool, device=joints.device)
        for b in range(B):
            for t in range(L):
                s = slice(max(0, t - w), min(L, t + w + 1))
                ok = float(dist[b, s].max() - dist[b, s].min()) <= STANDING_DIST
                far = dist[b, s] >= STANDING_ANGLE_MIN_DIST
                if ok and int(far.sum()) >= 2:
                    th = theta[b, s][far]
                    diffs = (th - th[0] + 180.0) % 360.0 - 180.0
                    ok = float(diffs.max() - diffs.min()) <= STANDING_ANGLE
                ok = ok and float(r_l[b, s].max() - r_l[b, s].min()) <= STANDING_RADIUS
                ok = ok and float(r_r[b, s].max() - r_r[b, s].min()) <= STANDING_RADIUS
                rows[b, t] = ok and float(tilt[b, t]) <= STANDING_TILT
            # one-row gaps merged, then runs of at least STANDING_MIN_SECONDS
            lab = rows[b].long()
            gap = (lab[1:-1] == 0) & (lab[:-2] == 1) & (lab[2:] == 1)
            lab[1:-1][gap] = 1
            keep = torch.zeros_like(lab)
            start = None
            for t in range(L + 1):
                on = t < L and lab[t] == 1
                if on and start is None:
                    start = t
                elif not on and start is not None:
                    if (t - start) / fps >= STANDING_MIN_SECONDS:
                        keep[start:t] = 1
                    start = None
            rows[b] = keep.bool()
        stance = torch.stack([rows, rows], dim=-1)
    else:
        raise ValueError(f"unseen foot rule {rule!r}, expected one of {UNSEEN_FOOT_RULES}")
    return stance & ~lying[..., None]


def effective_static_logits(static_conf_logits, seen, stance):
    """(brawler fork) static logits with the unseen feet's contact from the rule, and the per-frame
    hold source (B, L, 2) int8 [left, right]: HOLD_NETWORK where the network's confidence of the
    foot's ankle or foot joint is above STATIC_CONF, HOLD_UNSEEN_RULE where the foot is unseen and the
    rule gives it contact (and the network does not), else HOLD_NONE.
    """
    logits = static_conf_logits.clone()
    by_rule = ~seen & stance
    source = torch.zeros(seen.shape, dtype=torch.int8, device=seen.device)
    for s, (a, f) in enumerate(((0, 1), (2, 3))):
        net = static_conf_logits[..., [a, f]].sigmoid().amax(dim=-1) > STATIC_CONF
        for j in (a, f):
            logits[..., j] = torch.where(
                by_rule[..., s], logits[..., j].clamp_min(UNSEEN_HOLD_LOGIT), logits[..., j]
            )
        source[..., s] = torch.where(
            net, HOLD_NETWORK, torch.where(by_rule[..., s], HOLD_UNSEEN_RULE, HOLD_NONE)
        ).to(torch.int8)
    return logits, source


def static_joint_targets(joints, static_conf, hard=False, side=False):
    """IK targets of the static joints: x[t] = c[t-1] * x[t-1] + (1 - c[t-1]) * fk[t], i.e. a
    static joint stays where it was in the previous frame, weighted by its static confidence c.
    Solved as a parallel prefix scan of the affine recursion x[t] = a[t] * x[t-1] + b[t] (a[0] = 0
    keeps the first frame). (brawler fork) hard: a confidence above STATIC_CONF counts as 1, so a
    joint predicted static is held exactly, not with 10-20 % of the network's motion per frame.

    side (brawler fork): a foot (ankle + foot joint, STATIC_JOINT_IDS order L_ankle, L_foot, R_ankle,
    R_foot) whose larger confidence is above STATIC_CONF and whose ankle confidence is at least
    SIDE_ANKLE_MIN is held as a whole (both weights 1); implies hard.

    joints: (B, L, J, 3) the static joints' positions; static_conf: (B, L, J) -> (B, L, J, 3)
    """
    if side:
        static_conf = static_conf.clone()
        for a in (0, 2):  # ankle index of each foot; a + 1 is its foot joint
            pair = static_conf[..., a : a + 2]
            held = (pair.amax(dim=-1) > STATIC_CONF) & (static_conf[..., a] >= SIDE_ANKLE_MIN)
            static_conf[..., a : a + 2] = torch.where(held[..., None], torch.ones_like(pair), pair)
        hard = True
    if hard:
        static_conf = torch.where(static_conf > STATIC_CONF, torch.ones_like(static_conf), static_conf)
    a = torch.cat(
        [torch.zeros_like(static_conf[:, :1, :, None]), static_conf[:, :-1, :, None]], dim=1
    )
    b = joints * (1 - a)
    offset = 1
    while offset < b.shape[1]:
        b = torch.cat([b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]], dim=1)
        a = torch.cat([a[:, :offset], a[:, offset:] * a[:, :-offset]], dim=1)
        offset *= 2
    return b


def static_joint_ik(smplx_global, static_conf_logits, decoder, hard=False, side=False):
    """Inverse kinematics so that ankles, feet and wrists predicted static stay put.

    hard (brawler fork): hold joints with a static confidence above STATIC_CONF exactly, see
    static_joint_targets. Returns the post-processed (B, L, 63) world-frame body pose (axis-angle).
    """
    static_conf = static_conf_logits.sigmoid()  # (B, L, J)
    post_w_j3d, local_mat, _ = decoder.fk(**smplx_global, get_intermediate=True)

    post_target_j3d = post_w_j3d.clone()
    post_target_j3d[:, :, STATIC_JOINT_IDS] = static_joint_targets(
        post_w_j3d[:, :, STATIC_JOINT_IDS], static_conf, hard, side
    )
    # a toe predicted static can lie far from where it was when its contact began; pulling it all
    # the way back turns the ankle a lot (tens of degrees), against the network's ankle. Bounded
    # per axis like FlowHMR's ankle targets (github.com/flowhmr/flowhmr, runtime_v2m_postprocess)
    toes = [10, 11]
    offset = post_target_j3d[:, :, toes] - post_w_j3d[:, :, toes]
    bound = SIDE_TOE_MAX_OFFSET if side else TOE_TARGET_MAX_OFFSET  # (brawler fork) side_static_hold
    offset = offset.clamp(-bound, bound)
    post_target_j3d[:, :, toes] = post_w_j3d[:, :, toes] + offset

    parents = decoder.parents
    chains = [  # (chain, targets: joint ids, their indices within the chain)
        ([0, 1, 4, 7, 10], [7, 10], [3, 4]),  # left leg: ankle and foot
        ([0, 2, 5, 8, 11], [8, 11], [3, 4]),  # right leg
        ([9, 13, 16, 18, 20], [20], [4]),  # left arm: wrist
        ([9, 14, 17, 19, 21], [21], [4]),  # right arm
    ]
    for chain, joint_ids, target_ind in chains:
        solver = CCDIK(local_mat, parents, target_ind, post_target_j3d[:, :, joint_ids], chain)
        chain_rotmat = kinematics.get_rotation(solver.solve())
        local_mat = local_mat.clone()
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]

    body_pose = matrix_to_axis_angle(kinematics.get_rotation(local_mat[:, :, 1:]))
    return body_pose.flatten(2)  # (B, L, 63)
