"""ExMR: initial whole-body estimate, then FootMR and HandMR refine the extremities."""

import torch
import torch.nn as nn
from safetensors.torch import load_file

from exmr.body_models.smplx_model import SmplxModel
from exmr.geometry.camera import cliff_bbox_info, full_cam_transl, normalize_kp2d
from exmr.geometry.rotation_conversions import (
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from exmr.geometry.smoothing import gaussian_smooth, gaussian_smooth_rotations
from exmr.model import global_traj
from exmr.model.body_net import HAND_POSE, BodyNet
from exmr.model.decoder import Decoder
from exmr.model.refiners import FootRefiner, HandRefiner

# indices into the 22 SMPL-X body joints
KNEES, ANKLES = slice(4, 6), slice(7, 9)
SHOULDERS, ELBOWS, WRISTS = slice(16, 18), slice(18, 20), slice(20, 22)
# the same joints in the 63-dim body_pose (axis-angle of joints 1..21)
BODY_POSE_ANKLES, BODY_POSE_ELBOWS, BODY_POSE_WRISTS = slice(18, 24), slice(51, 57), slice(57, 63)
FOOT_KP2D = slice(15, 23)  # ankles + 6 foot keypoints of the COCO-23 layout
# temporal smoothing of the output motion: Gaussian sigma in seconds, one frame at 30 fps
SMOOTHING_SIGMA = 1 / 30


def _rot6d(rotmat):
    """(B, L, N, 3, 3) -> (B, L, N*6)"""
    return matrix_to_rotation_6d(rotmat).flatten(-2)


def _smooth_motion(smplx_params, sigma):
    """Gaussian smoothing over time (sigma in frames) of SMPL-X parameters, each (B, L, D):
    the rotations on SO(3) and the translation (the shape is the same in all frames)."""
    smoothed = dict(smplx_params)
    for key in ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose"):
        smoothed[key] = gaussian_smooth_rotations(smplx_params[key], sigma, dim=-2)
    smoothed["transl"] = gaussian_smooth(smplx_params["transl"], sigma, dim=-2)
    return smoothed


class ExMR(nn.Module):
    """Reconstructs SMPL-X body, hand and foot motion of one person from per-frame evidence.

    Inputs come from exmr.preprocessing: 2D keypoints, bounding boxes, camera intrinsics and
    angular velocity, SAM 3D Body and HaMeR image features. Outputs are SMPL-X parameters in the
    camera frame and in a gravity-aligned world frame.
    """

    def __init__(self, smplx_npz):
        super().__init__()
        self.body_net = BodyNet()
        self.foot_refiner = FootRefiner()
        self.hand_refiner = HandRefiner()
        self.decoder = Decoder(smplx_npz)
        # the SMPL-X mesh, to put the world-frame motion on the ground
        self.body_model = SmplxModel(smplx_npz)
        # camera angular velocity normalization (6D), loaded with the checkpoint
        self.register_buffer("cam_angvel_mean", torch.zeros(6))
        self.register_buffer("cam_angvel_std", torch.ones(6))

    @classmethod
    def from_pretrained(cls, checkpoint, smplx_npz, device="cuda"):
        """Load the weights from a released checkpoint (.safetensors)."""
        model = cls(smplx_npz)
        state_dict = load_file(str(checkpoint))
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()

    @torch.inference_mode()
    def forward(
        self,
        kp2d,
        bbx_xys,
        K_fullimg,
        cam_angvel,
        body_feat,
        hand_feat,
        length=None,
        static_cam=False,
        postprocess=True,
        snap_to_ground=False,
        smooth_incam=False,
        smooth_global=True,
        fps=30.0,
        up_from_camera=False,
        hard_static_hold=False,
        side_static_hold=False,
        unseen_foot_rule="none",
    ):
        """
        Args:
            kp2d: (B, L, 23, 3) COCO-23 keypoints (body + feet) in pixels, [x, y, conf]
            bbx_xys: (B, L, 3) person bounding box (center x, center y, size) in pixels
            K_fullimg: (B, L, 3, 3) camera intrinsics
            cam_angvel: (B, L, 6) camera angular velocity, see exmr.preprocessing
            body_feat: (B, L, 1024) SAM 3D Body image features
            hand_feat: (B, L, 2, 1024) HaMeR image features (left, right)
            length: (B,) valid length of each sequence, default: all L frames. Padded sequences
                (length < L) need postprocess, snap_to_ground, smooth_incam and smooth_global
                False: they work on whole sequences, padding included
            static_cam: whether the camera is known to be static (better world trajectory)
            postprocess: apply static-joint post-processing to the world-frame outputs and put
                them on the ground (the lowest mesh vertex over all frames at y=0)
            snap_to_ground: the ground is flat and there is nothing to stand on: keep the
                world-frame body on the ground (removes vertical drift), see
                global_traj.snap_to_ground
            smooth_incam: smooth the camera-frame motion over time (Gaussian, SMOOTHING_SIGMA);
                smoother, but follows fast motion, e.g. of the feet, slightly less closely in
                the image
            smooth_global: smooth the world-frame motion over time (Gaussian, SMOOTHING_SIGMA)
            fps: frame rate of the video, for the time windows of snap_to_ground, of the
                static-camera drift correction and of the smoothing
            up_from_camera: the world frame's gravity is estimated over the first
                global_traj.UP_INIT_SECONDS and carried by the camera rotation (cam_angvel),
                instead of the network's per-frame estimate, which takes a rolling camera for a
                leaning body (brawler fork, see global_traj.up_from_camera_gv)
            hard_static_hold: with postprocess, joints predicted static with a confidence above
                global_traj.STATIC_CONF are held exactly by the IK instead of with the weight of
                their confidence (brawler fork, see global_traj.static_joint_targets)
            side_static_hold: with postprocess, a foot predicted static is held as a whole (ankle and
                foot joint) when its ankle confidence is at least global_traj.SIDE_ANKLE_MIN, with a
                looser toe bound (brawler fork; implies hard_static_hold)
            unseen_foot_rule: "none" (default, as upstream) or a rule of
                global_traj.UNSEEN_FOOT_RULES: a foot whose 2D keypoints are all below
                global_traj.FOOT_SEEN_CONF (outside the frame or hidden) gets its contact from the rule
                on the body instead of the network's static confidence (brawler fork)
        Returns:
            dict with
              smplx_incam: SMPL-X parameters in the camera frame (each (B, L, D))
              smplx_global: SMPL-X parameters in the gravity-aligned world frame (y up, ground
                  at y=0 with postprocess or snap_to_ground; with postprocess, the motion starts
                  at x=z=0 facing +z)
              static_conf_logits: (B, L, 6) logits of being static (not moving) for
                  [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
              up_correction_deg: (B, L) with up_from_camera: the angle between the network's
                  and the carried gravity direction per frame (otherwise zeros)
              smplx_global_raw: smplx_global before the post-processing (brawler fork)
              static_conf_logits_effective, foot_seen, foot_rule_stance, foot_hold_source,
                  foot_ik_shift: per-frame flags of the post-processing, see postprocess_world
                  (brawler fork)
        """
        B, L = kp2d.shape[:2]
        if length is None:
            length = torch.full((B,), L, device=kp2d.device)
        elif any([postprocess, snap_to_ground, smooth_incam, smooth_global]) and bool(
            (length < L).any()
        ):
            raise ValueError(
                "post-processing and smoothing (postprocess, snap_to_ground, smooth_incam, "
                "smooth_global) need unpadded sequences: run padded sequences one at a time, or "
                "with all of them disabled"
            )
        obs = normalize_kp2d(kp2d, bbx_xys)
        cliffcam = cliff_bbox_info(bbx_xys, K_fullimg)
        cam_angvel_norm = (cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std

        # 1. initial whole-body estimate
        body_out = self.body_net(length, obs, cliffcam, cam_angvel_norm, body_feat, hand_feat)
        params = self.decoder.decode(body_out["x"])
        smplx_incam = {
            "body_pose": params["body_pose"],
            "betas": params["betas"],
            "global_orient": params["global_orient"],
            "left_hand_pose": params["left_hand_pose"],
            "right_hand_pose": params["right_hand_pose"],
            "transl": full_cam_transl(body_out["pred_cam"], bbx_xys, K_fullimg),
        }
        body_pose = smplx_incam["body_pose"]  # refined in place below

        # 2. FootMR: refine the global ankle rotations, then express them relative to the knee
        fk_rot = self.decoder.fk_rot(body_pose, smplx_incam["global_orient"])  # (B, L, 22, 3, 3)
        knee_rot = fk_rot[:, :, KNEES]
        foot_rot6d = torch.cat([_rot6d(knee_rot), _rot6d(fk_rot[:, :, ANKLES])], dim=-1)
        ankle_rot = self.foot_refiner(
            length, obs[:, :, FOOT_KP2D].clone(), cliffcam.clone(), foot_rot6d
        )
        ankle_rot = rotation_6d_to_matrix(ankle_rot.reshape(B, L, -1, 6))  # (B, L, 2, 3, 3)
        body_pose[:, :, BODY_POSE_ANKLES] = matrix_to_axis_angle(knee_rot.mT @ ankle_rot).flatten(
            -2
        )

        # 3. HandMR: refine global elbows and wrists and the finger rotations
        fk_rot = self.decoder.fk_rot(body_pose, smplx_incam["global_orient"])
        hand_pose_r6d = self.decoder.denormalize(body_out["x"][..., HAND_POSE], HAND_POSE)
        hand_rot6d = torch.cat(
            [_rot6d(fk_rot[:, :, ELBOWS]), _rot6d(fk_rot[:, :, WRISTS]), hand_pose_r6d], dim=-1
        )
        refined = self.hand_refiner(length, hand_feat, cliffcam.clone(), hand_rot6d)
        refined = rotation_6d_to_matrix(refined.reshape(B, L, -1, 6))  # (B, L, 34, 3, 3)
        elbow_rot, wrist_rot, finger_rot = refined[:, :, 0:2], refined[:, :, 2:4], refined[:, :, 4:]
        # global -> parent-relative: elbow relative to the shoulder, wrist to the refined elbow
        shoulder_rot = fk_rot[:, :, SHOULDERS]
        body_pose[:, :, BODY_POSE_ELBOWS] = matrix_to_axis_angle(
            shoulder_rot.mT @ elbow_rot
        ).flatten(-2)
        body_pose[:, :, BODY_POSE_WRISTS] = matrix_to_axis_angle(elbow_rot.mT @ wrist_rot).flatten(
            -2
        )
        finger_aa = matrix_to_axis_angle(finger_rot).flatten(-2)  # (B, L, 90)
        smplx_incam["left_hand_pose"] = finger_aa[:, :, :45]
        smplx_incam["right_hand_pose"] = finger_aa[:, :, 45:]

        # 4. world frame: same articulation, rolled-out root trajectory
        global_orient_gv = params["global_orient_gv"]
        up_correction = torch.zeros(B, L, device=global_orient_gv.device)
        if up_from_camera:
            global_orient_gv, up_correction = global_traj.up_from_camera_gv(
                global_orient_gv,
                smplx_incam["global_orient"],
                cam_angvel,
                round(global_traj.UP_INIT_SECONDS * fps),
            )
        smplx_global = {
            "body_pose": body_pose,
            "betas": smplx_incam["betas"],
            "left_hand_pose": smplx_incam["left_hand_pose"],
            "right_hand_pose": smplx_incam["right_hand_pose"],
            **global_traj.rollout_global_params(
                global_orient_gv,
                params["local_transl_vel"],
                smplx_incam["global_orient"],
                cam_angvel,
            ),
        }
        static_conf_logits = body_out["static_conf_logits"]
        # (brawler fork) the world-frame motion before the post-processing, returned so that the
        # post-processing (postprocess_world) can be re-run without the network
        smplx_global_raw = {k: v.clone() for k, v in smplx_global.items()}
        foot_seen = global_traj.feet_seen(kp2d)
        smplx_global, flags = postprocess_world(
            smplx_global, smplx_incam, static_conf_logits, foot_seen, self.decoder,
            self.body_model, fps, postprocess=postprocess, static_cam=static_cam,
            snap_to_ground=snap_to_ground, smooth_global=smooth_global,
            hard_static_hold=hard_static_hold, side_static_hold=side_static_hold,
            unseen_foot_rule=unseen_foot_rule,
        )
        if smooth_incam:
            smplx_incam = _smooth_motion(smplx_incam, SMOOTHING_SIGMA * fps)

        return {
            "smplx_incam": smplx_incam,
            "smplx_global": smplx_global,
            "static_conf_logits": static_conf_logits,
            "up_correction_deg": up_correction,
            "smplx_global_raw": smplx_global_raw,
            **flags,
        }


def postprocess_world(
    smplx_global,
    smplx_incam,
    static_conf_logits,
    foot_seen,
    decoder,
    body_model,
    fps,
    postprocess=True,
    static_cam=False,
    snap_to_ground=False,
    smooth_global=True,
    hard_static_hold=False,
    side_static_hold=False,
    unseen_foot_rule="none",
):
    """Steps 4b-5 of ExMR.forward on the world-frame motion (brawler fork: one function, so that the
    post-processing can be re-run without the network on a saved smplx_global_raw).

    smplx_global: the rolled-out world-frame SMPL-X params (smplx_global_raw); smplx_incam: the
    camera-frame params before any incam smoothing; static_conf_logits: the network's (B, L, 6);
    foot_seen: (B, L, 2) bool, global_traj.feet_seen of the 2D keypoints.
    unseen_foot_rule: "none" (as upstream) or a rule of global_traj.UNSEEN_FOOT_RULES giving
    unseen feet their contact; it acts through the same static logits as the network's own
    contacts, i.e. on the root translation correction, the IK and the contacts.
    Returns the post-processed smplx_global and per-frame flags:
      static_conf_logits_effective (B, L, 6): the logits the post-processing used;
      foot_seen (B, L, 2) bool; foot_rule_stance (B, L, 2) bool (False with rule "none");
      foot_hold_source (B, L, 2) int8: global_traj.HOLD_NONE / HOLD_NETWORK / HOLD_UNSEEN_RULE;
      foot_ik_shift (B, L, 2): how far (m) the IK moved each ankle (0 without postprocess).
    """
    smplx_global = dict(smplx_global)
    B, L = static_conf_logits.shape[:2]
    if unseen_foot_rule != "none":
        joints = decoder.fk(**smplx_global)
        stance = global_traj.unseen_stance(joints, fps, unseen_foot_rule)
    else:
        stance = torch.zeros_like(foot_seen)
    logits, source = global_traj.effective_static_logits(static_conf_logits, foot_seen, stance)
    ik_shift = torch.zeros(B, L, 2, device=static_conf_logits.device)
    if postprocess:
        if static_cam:
            smplx_global["transl"] = global_traj.static_cam_transl(
                smplx_incam, smplx_global, logits, decoder, fps
            )
        else:
            smplx_global["transl"] = global_traj.static_joint_transl(smplx_global, logits, decoder)
    # the snap: after the static-joint translation correction, which would cancel it where a
    # joint is static, and before the IK, which then keeps static joints in place on the result
    if snap_to_ground:
        smplx_global["transl"] = global_traj.snap_to_ground(smplx_global, body_model, fps)
    if postprocess:
        before = decoder.fk(**smplx_global)[:, :, 7:9]
        smplx_global["body_pose"] = global_traj.static_joint_ik(
            smplx_global, logits, decoder, hard=hard_static_hold, side=side_static_hold,
        )
        ik_shift = (decoder.fk(**smplx_global)[:, :, 7:9] - before).norm(dim=-1)

    # 5. temporal smoothing, after the IK so that it also smooths the IK's changes
    if smooth_global:
        smplx_global = _smooth_motion(smplx_global, SMOOTHING_SIGMA * fps)
    if postprocess and not snap_to_ground:
        # one shift for all frames, so last: puts the final mesh on the ground
        smplx_global["transl"] = global_traj.put_on_ground(smplx_global, body_model)
    if postprocess:
        # facing +z in the first frame, as BVH, Blender and Unity expect a character to start
        smplx_global["global_orient"], smplx_global["transl"] = global_traj.face_forward(
            smplx_global, decoder
        )
    flags = {
        "static_conf_logits_effective": logits,
        "foot_seen": foot_seen,
        "foot_rule_stance": stance,
        "foot_hold_source": source,
        "foot_ik_shift": ik_shift,
    }
    return smplx_global, flags
