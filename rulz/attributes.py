"""Face attributes: expression labels, head pose estimation and how search terms map onto them.

Head pose comes from YuNet's 5 landmarks (eyes, nose tip, mouth corners), measured in a frame
aligned with the eye line:
  yaw   = horizontal nose offset from the eye midpoint / eye distance
          (negative = the face is turned towards the LEFT side of the photo)
  pitch = nose height between eye line (0) and mouth line (1); ~0.57 when level,
          lower = looking up, higher = looking down
  roll  = eye-line angle in degrees (head tilt)
"""
from __future__ import annotations

import math
import re

import numpy as np

# FER+ output order -> stored labels
EXPRESSIONS = ["neutral", "happy", "surprised", "sad", "angry", "disgusted", "scared", "contempt"]
EXPRESSION_DISPLAY = {
    "neutral": "Neutral", "happy": "Smiling", "surprised": "Surprised", "sad": "Sad", "angry": "Angry",
    "disgusted": "Disgusted", "scared": "Scared", "contempt": "Smirking",
}

YAW_TURNED = 0.20      # ~15-20 degrees
YAW_PROFILE = 0.60     # ~45 degrees or more
PITCH_UP = 0.40
PITCH_DOWN = 0.74
ROLL_TILTED = 15.0


def head_pose(landmarks: np.ndarray) -> tuple[float, float, float]:
    """landmarks: 5x2 (right eye, left eye, nose, right mouth, left mouth) in pixels."""
    lm = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    re_, le, nose, mr, ml = lm
    roll = math.degrees(math.atan2(le[1] - re_[1], le[0] - re_[0]))
    a = math.radians(-roll)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q = (lm - (re_ + le) / 2) @ rot.T
    eye_dist = max(float(np.linalg.norm(le - re_)), 1e-6)
    yaw = q[2, 0] / eye_dist
    mouth_y = (q[3, 1] + q[4, 1]) / 2
    pitch = q[2, 1] / mouth_y if mouth_y > 1e-6 else 0.57
    return float(yaw), float(pitch), float(roll)


def describe(expression: str | None, yaw: float | None, pitch: float | None, roll: float | None) -> list[str]:
    """Short human-readable tags for a face, e.g. ['Smiling', 'Facing left']."""
    out = []
    if expression:
        out.append(EXPRESSION_DISPLAY.get(expression, expression.title()))
    if yaw is not None:
        if abs(yaw) >= YAW_PROFILE:
            out.append("Side profile, " + ("facing left" if yaw < 0 else "facing right"))
        elif yaw <= -YAW_TURNED:
            out.append("Facing left")
        elif yaw >= YAW_TURNED:
            out.append("Facing right")
        elif pitch is not None and PITCH_UP <= pitch <= PITCH_DOWN:
            out.append("Facing camera")
    if pitch is not None:
        if pitch < PITCH_UP:
            out.append("Looking up")
        elif pitch > PITCH_DOWN:
            out.append("Looking down")
    if roll is not None and abs(roll) >= ROLL_TILTED:
        out.append("Head tilted")
    return out


# ---- search vocabulary ------------------------------------------------------------
# phrase -> SQL condition on the faces table. Longer phrases are matched first.
_EXPR = lambda e: f"expression='{e}'"  # noqa: E731
SEARCH_TERMS: dict[str, str] = {
    "smiling": _EXPR("happy"), "smile": _EXPR("happy"), "smiles": _EXPR("happy"), "happy": _EXPR("happy"),
    "laughing": _EXPR("happy"), "grinning": _EXPR("happy"),
    "neutral": _EXPR("neutral"), "serious": _EXPR("neutral"), "not smiling": "expression<>'happy'",
    "straight face": _EXPR("neutral"),
    "surprised": _EXPR("surprised"), "shocked": _EXPR("surprised"), "surprise": _EXPR("surprised"),
    "sad": _EXPR("sad"), "unhappy": _EXPR("sad"), "crying": _EXPR("sad"),
    "angry": _EXPR("angry"), "mad": _EXPR("angry"), "annoyed": _EXPR("angry"),
    "disgusted": _EXPR("disgusted"), "grossed out": _EXPR("disgusted"),
    "scared": _EXPR("scared"), "afraid": _EXPR("scared"), "fearful": _EXPR("scared"), "frightened": _EXPR("scared"),
    "smirking": _EXPR("contempt"), "smirk": _EXPR("contempt"), "contempt": _EXPR("contempt"),

    "facing left": f"yaw<=-{YAW_TURNED}", "looking left": f"yaw<=-{YAW_TURNED}", "turned left": f"yaw<=-{YAW_TURNED}",
    "facing right": f"yaw>={YAW_TURNED}", "looking right": f"yaw>={YAW_TURNED}", "turned right": f"yaw>={YAW_TURNED}",
    "facing camera": f"abs(yaw)<{YAW_TURNED} AND pitch BETWEEN {PITCH_UP} AND {PITCH_DOWN}",
    "looking at camera": f"abs(yaw)<{YAW_TURNED} AND pitch BETWEEN {PITCH_UP} AND {PITCH_DOWN}",
    "facing forward": f"abs(yaw)<{YAW_TURNED} AND pitch BETWEEN {PITCH_UP} AND {PITCH_DOWN}",
    "facing the camera": f"abs(yaw)<{YAW_TURNED} AND pitch BETWEEN {PITCH_UP} AND {PITCH_DOWN}",
    "front": f"abs(yaw)<{YAW_TURNED} AND pitch BETWEEN {PITCH_UP} AND {PITCH_DOWN}",
    "side profile": f"abs(yaw)>={YAW_PROFILE}", "profile": f"abs(yaw)>={YAW_PROFILE}",
    "side view": f"abs(yaw)>={YAW_PROFILE}", "turned away": f"abs(yaw)>={YAW_TURNED}",
    "looking up": f"pitch<{PITCH_UP}", "facing up": f"pitch<{PITCH_UP}",
    "looking down": f"pitch>{PITCH_DOWN}", "facing down": f"pitch>{PITCH_DOWN}",
    "head tilted": f"abs(roll)>={ROLL_TILTED}", "tilted": f"abs(roll)>={ROLL_TILTED}",
    "head tilt": f"abs(roll)>={ROLL_TILTED}",
}
_PHRASES = sorted(SEARCH_TERMS, key=len, reverse=True)
_FILLER = {"with", "who", "is", "are", "was", "people", "person", "face", "faces", "photos", "photo", "pictures",
           "of", "the", "a", "an", "where", "everyone", "someone", "anyone", "in", "at"}


def parse_group(text: str) -> tuple[str, list[str], list[str]]:
    """Split one search group into (name text, SQL conditions, matched attribute phrases)."""
    s = " " + re.sub(r"\s+", " ", text.lower()).strip() + " "
    conds, matched = [], []
    for phrase in _PHRASES:
        pat = f" {phrase} "
        while pat in s:
            s = s.replace(pat, " ", 1)
            if SEARCH_TERMS[phrase] not in conds:
                conds.append(SEARCH_TERMS[phrase])
            matched.append(phrase)
    words = [w for w in s.split() if w not in _FILLER]
    return " ".join(words), conds, matched
