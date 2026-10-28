"""Body pose (standing / sitting / lying down / arms raised) from MediaPipe BlazePose via OpenCV DNN.

Two models from the OpenCV Model Zoo (Apache 2.0):
  person_detection_mediapipe_2023mar.onnx  finds people (a head box + hip/shoulder alignment points)
  pose_estimation_mediapipe_2023mar.onnx   33 body landmarks per person, in the image and in 3D metres

Pre/post-processing is adapted from opencv_zoo's mp_persondet.py and mp_pose.py (Apache 2.0).
Poses are estimated per detected face, so they can be searched per person.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import config

# BlazePose landmark indices
NOSE, L_EYE, R_EYE = 0, 2, 5
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28

POSES = ("standing", "sitting", "lying")  # display names live in attributes.POSE_DISPLAY

# Joint-angle rules (degrees, measured on the 3D landmarks so they don't depend on camera angle).
SIT_HIP_MAX = 130       # torso-to-thigh angle; ~90 when seated, ~175 when upright
STAND_HIP_MIN = 145
STAND_KNEE_MIN = 130
LYING_TILT_MIN = 55     # torso angle from vertical in the photo
UPSIDE_DOWN_TILT = 130  # beyond this the "body" is a reflection or a misdetection
LYING_KNEES_ABOVE = -0.2  # thigh drop below this (knees above hips) + leaning torso = lying
SIT_THIGH_DROP_MAX = 0.45   # 2D: how far knees sit below hips, relative to torso length
STAND_THIGH_DROP_MIN = 0.6
ANKLE_VISIBLE = 0.6     # ankle visibility needed before sitting/standing is decided
MAX_FACES = 12          # pose is estimated for at most this many (largest) faces per photo


@dataclass
class Body:
    head: tuple[float, float]       # pixel position of the nose/eyes (used to match a face)
    pose: str | None                # one of POSES, or None when the legs/torso aren't visible enough
    arms_up: bool


def _ssd_anchors() -> np.ndarray:
    """MediaPipe pose-detection anchors: 224x224 input, strides 8/16/32/32/32, 2 anchors per layer."""
    out = []
    strides = [8, 16, 32, 32, 32]
    i = 0
    while i < len(strides):
        stride, per_cell = strides[i], 0
        while i < len(strides) and strides[i] == stride:  # layers with equal stride share a grid
            per_cell += 2
            i += 1
        n = math.ceil(224 / stride)
        for y in range(n):
            for x in range(n):
                out += [((x + 0.5) / n, (y + 0.5) / n)] * per_cell
    return np.array(out, dtype=np.float32)


class BodyAnalyzer:
    """Not thread-safe: create one per thread."""

    DET_SIZE = 224
    POSE_SIZE = 256

    def __init__(self, score_threshold: float = 0.5, pose_threshold: float = 0.5):
        self.score_threshold = score_threshold
        self.pose_threshold = pose_threshold
        self.detector = cv2.dnn.readNet(str(config.PERSON_DETECTOR_MODEL))
        self.landmarker = cv2.dnn.readNet(str(config.POSE_MODEL))
        self.anchors = _ssd_anchors()

    @staticmethod
    def available() -> bool:
        return config.PERSON_DETECTOR_MODEL.exists() and config.POSE_MODEL.exists()

    # ---- person detection -----------------------------------------------------
    def _detect_people(self, bgr: np.ndarray) -> np.ndarray:
        """Rows of [face box x1,y1,x2,y2, 4 alignment points (hip, full body, shoulder, upper body), score]."""
        h, w = bgr.shape[:2]
        size = self.DET_SIZE
        ratio = min(size / h, size / w)
        nh, nw = int(h * ratio), int(w * ratio)
        img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (nw, nh)).astype(np.float32) / 127.5 - 1.0
        top, left = (size - nh) // 2, (size - nw) // 2
        img = cv2.copyMakeBorder(img, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT, value=0)
        self.detector.setInput(np.transpose(img, (2, 0, 1))[np.newaxis])
        outs = self.detector.forward(self.detector.getUnconnectedOutLayersNames())
        # Pick outputs by shape: their order differs between OpenCV versions.
        scores = next(o for o in outs if o.shape[-1] == 1)[0, :, 0].astype(np.float64)
        deltas = next(o for o in outs if o.shape[-1] == 12)[0]
        scores = 1 / (1 + np.exp(-np.clip(scores, -100, 100)))
        scale = max(w, h)
        pad = np.array([left, top], np.float32) / ratio
        cxy = deltas[:, :2] / size + self.anchors
        wh = deltas[:, 2:4] / size
        boxes = np.concatenate([(cxy - wh / 2) * scale - pad, (cxy + wh / 2) * scale - pad], axis=1)
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.score_threshold, 0.3)
        if len(keep) == 0:
            return np.empty((0, 13), np.float32)
        keep = np.asarray(keep).reshape(-1)
        points = (deltas[keep, 4:].reshape(-1, 4, 2) / size + self.anchors[keep][:, None, :]) * scale - pad
        return np.c_[boxes[keep], points.reshape(-1, 8), scores[keep]]

    # ---- landmarks -------------------------------------------------------------
    def _landmarks(self, bgr: np.ndarray, person: np.ndarray):
        """Return (image landmarks [39,5]: x, y, z, visibility, presence; world landmarks [39,3]) or None."""
        hip = person[4:6].astype(np.float64).copy()
        full = person[6:8].astype(np.float64)
        dist = float(np.linalg.norm(hip - full))
        if dist < 4:
            return None
        box = np.array([hip - dist, hip + dist], np.int32)
        h, w = bgr.shape[:2]
        clipped = box.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, w)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, h)
        crop = bgr[clipped[0][1]:clipped[1][1], clipped[0][0]:clipped[1][0]]
        if crop.size == 0:
            return None
        left, top = clipped[0] - box[0]
        right, bottom = box[1] - clipped[1]
        crop = cv2.copyMakeBorder(crop, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=0)
        pad_bias = clipped[0] - [left, top]
        hip_c = hip - pad_bias
        full_c = full - pad_bias
        radians = np.pi / 2 - np.arctan2(-(full_c[1] - hip_c[1]), full_c[0] - hip_c[0])
        radians -= 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = float(np.rad2deg(radians))
        rot = cv2.getRotationMatrix2D((float(hip_c[0]), float(hip_c[1])), angle, 1.0)
        rotated = cv2.warpAffine(crop, rot, (crop.shape[1], crop.shape[0]))
        blob = cv2.cvtColor(cv2.resize(rotated, (self.POSE_SIZE, self.POSE_SIZE), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.landmarker.setInput(blob[np.newaxis])
        outs = self.landmarker.forward(self.landmarker.getUnconnectedOutLayersNames())
        by_size = {o.size: o for o in outs}
        conf, lm, world = float(by_size[1].reshape(-1)[0]), by_size[195], by_size[117]
        if conf < self.pose_threshold:
            return None
        lm = lm.reshape(-1, 5).astype(np.float64)
        world = world.reshape(-1, 3).astype(np.float64)
        lm[:, 3:] = 1 / (1 + np.exp(-lm[:, 3:]))
        crop_wh = np.array([crop.shape[1], crop.shape[0]], np.float64)
        lm[:, :2] = (lm[:, :2] - self.POSE_SIZE / 2) * (crop_wh / self.POSE_SIZE)
        undo = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
        lm[:, :2] = lm[:, :2] @ undo
        world[:, :2] = world[:, :2] @ undo
        # The crop center, mapped back through the inverse rotation, plus the crop offset.
        rc, t = rot[:, :2].T, rot[:, 2]
        inv = np.c_[rc, [-rc[0] @ t, -rc[1] @ t]]
        center = np.append(crop_wh / 2, 1)
        lm[:, :2] += np.array([center @ inv[0], center @ inv[1]]) + pad_bias
        return lm, world

    # ---- public ------------------------------------------------------------------
    def analyze(self, bgr: np.ndarray, faces_px: list[tuple[float, float, float, float]]) -> list[Body | None]:
        """Pose for each face box (x, y, w, h in pixels), or None where no body could be read.

        The person detector shrinks its input to 224 px and finds people by their face, so in full-body
        photos it misses small faces. Running it on a crop around each known face fixes that, and also
        ties each body to its face.
        """
        h, w = bgr.shape[:2]
        out: list[Body | None] = [None] * len(faces_px)
        order = sorted(range(len(faces_px)), key=lambda i: -faces_px[i][2] * faces_px[i][3])[:MAX_FACES]
        for i in order:
            fx, fy, fw, fh = faces_px[i]
            x0, x1 = int(max(0, fx + fw / 2 - fw * 4)), int(min(w, fx + fw / 2 + fw * 4))
            y0, y1 = int(max(0, fy - fh * 1.5)), int(min(h, fy + fh * 9))
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            people = self._detect_people(np.ascontiguousarray(bgr[y0:y1, x0:x1]))
            if not len(people):
                continue
            people = people.copy()
            people[:, 0:12:2] += x0
            people[:, 1:12:2] += y0
            fcx, fcy = fx + fw / 2, fy + fh / 2
            dist = np.hypot((people[:, 0] + people[:, 2]) / 2 - fcx, (people[:, 1] + people[:, 3]) / 2 - fcy)
            j = int(np.argmin(dist))
            if dist[j] > max(fw, fh):  # the detected person isn't this face
                continue
            result = self._landmarks(bgr, people[j])
            if result is None:
                continue
            lm, world = result
            head = _head_point(lm, w, h) or (fcx, fcy)
            if abs(head[0] - fcx) > fw * 1.2 or abs(head[1] - fcy) > fh * 1.2:
                continue  # landmarks drifted onto someone else
            pose, arms_up = classify(lm, world, w, h)
            out[i] = Body(head, pose, arms_up)
        return out


def _usable(lm: np.ndarray, i: int, w: int, h: int) -> bool:
    x, y, _, vis, presence = lm[i]
    return vis > 0.5 and presence > 0.5 and -0.02 * w <= x <= 1.02 * w and -0.02 * h <= y <= 1.02 * h


def _head_point(lm, w, h):
    for i in (NOSE,):
        if _usable(lm, i, w, h):
            return float(lm[i, 0]), float(lm[i, 1])
    eyes = [i for i in (L_EYE, R_EYE) if _usable(lm, i, w, h)]
    if eyes:
        return float(np.mean(lm[eyes, 0])), float(np.mean(lm[eyes, 1]))
    return None


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def classify(lm: np.ndarray, world: np.ndarray, w: int, h: int) -> tuple[str | None, bool]:
    """Pose label from 2D (image) and 3D (world) landmarks. Returns (pose or None, arms raised)."""
    ok = lambda i: _usable(lm, i, w, h)  # noqa: E731
    shoulders = [i for i in (L_SHOULDER, R_SHOULDER) if ok(i)]
    hips = [i for i in (L_HIP, R_HIP) if ok(i)]
    if not shoulders or not hips:
        return None, False
    mid_sh, mid_hip = lm[shoulders, :2].mean(0), lm[hips, :2].mean(0)
    torso = mid_sh - mid_hip
    torso_len = float(np.linalg.norm(torso))
    tilt = math.degrees(math.atan2(abs(torso[0]), -torso[1])) if torso_len > 1 else 0.0  # 0 = upright

    arms_up = False
    if tilt < 45:
        for wrist, shoulder in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
            if ok(wrist) and ok(shoulder) and lm[wrist, 1] < lm[shoulder, 1] - 0.25 * torso_len:
                arms_up = True

    if tilt >= UPSIDE_DOWN_TILT:
        return None, False
    if tilt >= LYING_TILT_MIN:
        return "lying", False

    hip_angles, knee_angles, drops = [], [], []
    for sh, hip, knee, ankle in ((L_SHOULDER, L_HIP, L_KNEE, L_ANKLE), (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE)):
        if not (ok(hip) and ok(knee)):
            continue
        sh_point = world[sh] if ok(sh) else world[[L_SHOULDER, R_SHOULDER]].mean(0)
        hip_angles.append(_angle(sh_point - world[hip], world[knee] - world[hip]))
        if torso_len > 1:
            drops.append(float(lm[knee, 1] - lm[hip, 1]) / torso_len)
        if ok(ankle):
            knee_angles.append(_angle(world[hip] - world[knee], world[ankle] - world[knee]))
    if not hip_angles:
        return None, arms_up  # legs not in the photo: can't tell
    hip_a = float(np.median(hip_angles))
    knee_a = float(np.median(knee_angles)) if knee_angles else None
    drop = float(np.median(drops)) if drops else None
    if drop is not None and drop < LYING_KNEES_ABOVE and tilt >= 40:
        return "lying", arms_up  # reclining on an elbow: torso leaning back, knees level with or above hips
    # In waist-up photos the model invents plausible legs (often a seated pose), so only judge
    # sitting/standing when a lower leg is genuinely in the photo.
    legs_visible = any(ok(ankle) and lm[ankle, 3] > ANKLE_VISIBLE and lm[ankle, 1] < 0.985 * h
                       for ankle in (L_ANKLE, R_ANKLE))
    if not legs_visible:
        return None, arms_up
    # 3D angles catch side views; the 2D thigh drop catches seated people facing the camera, whose
    # thighs point at the lens (foreshortened) and make the 3D hip angle look straighter than it is.
    if hip_a <= SIT_HIP_MAX or (drop is not None and drop < SIT_THIGH_DROP_MAX and tilt < 30):
        return "sitting", arms_up
    if (hip_a >= STAND_HIP_MIN and (knee_a is None or knee_a >= STAND_KNEE_MIN)
            and (drop is None or drop >= STAND_THIGH_DROP_MIN)):
        return "standing", arms_up
    return None, arms_up
