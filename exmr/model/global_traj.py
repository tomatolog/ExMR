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


def static_joint_targets(joints, static_conf, hard=False):
    """IK targets of the static joints: x[t] = c[t-1] * x[t-1] + (1 - c[t-1]) * fk[t], i.e. a
    static joint stays where it was in the previous frame, weighted by its static confidence c.
    Solved as a parallel prefix scan of the affine recursion x[t] = a[t] * x[t-1] + b[t] (a[0] = 0
    keeps the first frame). (brawler fork) hard: a confidence above STATIC_CONF counts as 1, so a
    joint predicted static is held exactly, not with 10-20 % of the network's motion per frame.

    joints: (B, L, J, 3) the static joints' positions; static_conf: (B, L, J) -> (B, L, J, 3)
    """
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


def static_joint_ik(smplx_global, static_conf_logits, decoder, hard=False):
    """Inverse kinematics so that ankles, feet and wrists predicted static stay put.

    hard (brawler fork): hold joints with a static confidence above STATIC_CONF exactly, see
    static_joint_targets. Returns the post-processed (B, L, 63) world-frame body pose (axis-angle).
    """
    static_conf = static_conf_logits.sigmoid()  # (B, L, J)
    post_w_j3d, local_mat, _ = decoder.fk(**smplx_global, get_intermediate=True)

    post_target_j3d = post_w_j3d.clone()
    post_target_j3d[:, :, STATIC_JOINT_IDS] = static_joint_targets(
        post_w_j3d[:, :, STATIC_JOINT_IDS], static_conf, hard
    )
    # a toe predicted static can lie far from where it was when its contact began; pulling it all
    # the way back turns the ankle a lot (tens of degrees), against the network's ankle. Bounded
    # per axis like FlowHMR's ankle targets (github.com/flowhmr/flowhmr, runtime_v2m_postprocess)
    toes = [10, 11]
    offset = post_target_j3d[:, :, toes] - post_w_j3d[:, :, toes]
    offset = offset.clamp(-TOE_TARGET_MAX_OFFSET, TOE_TARGET_MAX_OFFSET)
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
