"""Raw video -> the exact features the models were trained on.

A line-by-line port of the team's cache builder (notebooks/asl-shared-cache-building-id00.ipynb):
  * all frames decoded with OpenCV, BGR -> RGB
  * 16 frame indices: np.linspace(0, n-1, 16).astype(int); clips shorter than 16 frames repeat the last
  * MediaPipe Tasks 0.10.35, VIDEO mode: hand (num_hands=2, min_det 0.5), pose_landmarker_lite, face (1 face)
  * hands assigned left/right by screen x-position of the wrist (not MediaPipe's handedness label)
  * landmark vector = [left hand 21, right hand 21, pose 11-16, face brow 10] x (x, y),
    centred on the shoulder midpoint and divided by shoulder width (fallback centre 0.5, scale 1)
  * appearance = pose-skeleton bbox + 25% margin, cv2.resize to 112 x 112 (full frame if no pose)

The only deliberate difference: the cache builder held every decoded frame in memory; here frames
are counted first and only the 16 sampled ones are kept (identical output, bounded memory for uploads).
"""
from __future__ import annotations

import threading
import types
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

FRAME_COUNT = 16
IMG_SIZE = 112
SIGNER_BBOX_MARGIN = 0.25
N_HAND_LANDMARKS = 21
POSE_SUBSET_IDX = [11, 12, 13, 14, 15, 16]
FACE_SUBSET_IDX = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
LANDMARK_VECTOR_DIM = (N_HAND_LANDMARKS * 2 + len(POSE_SUBSET_IDX) + len(FACE_SUBSET_IDX)) * 2  # 116

MEDIAPIPE_DIR = Path(__file__).resolve().parent.parent / "assets" / "mediapipe"

# Upload guards (training clips: median ~73 frames / 2.5 s, max 244 frames at 640x480)
MAX_DURATION_S = 20.0
MAX_TOTAL_FRAMES = 1200
MIN_FRAMES = 2


class VideoError(ValueError):
    """Raised with a user-readable message when a video cannot be used."""


@dataclass
class VideoInfo:
    fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float
    sampled_indices: list = field(default_factory=list)


@dataclass
class ClipFeatures:
    landmarks: np.ndarray        # (16, 116) float32 -- same as cache["landmarks"]
    appearance: np.ndarray       # (16, 1, 112, 112, 3) uint8 RGB -- same as cache["appearance"]
    info: VideoInfo
    frames_with_hands: int = 0
    frames_with_pose: int = 0
    frames_with_face: int = 0


# ------------------------------------------------------------------ frame sampling
def sample_frame_indices(total_frames, target_count=FRAME_COUNT):
    if total_frames <= 0:
        return [0] * target_count
    if total_frames >= target_count:
        return np.linspace(0, total_frames - 1, target_count).astype(int).tolist()
    return list(range(total_frames)) + [total_frames - 1] * (target_count - total_frames)


def read_sampled_frames(video_path, max_duration_s=MAX_DURATION_S):
    """Two passes: count decodable frames, then keep only the sampled ones (RGB)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError("The file could not be opened as a video. Please upload an MP4, MOV, AVI or WEBM clip.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 30.0                     # same default as the notebooks
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total = 0
    while cap.grab():
        total += 1
        if total > MAX_TOTAL_FRAMES:
            cap.release()
            raise VideoError(f"The video is too long (over {MAX_TOTAL_FRAMES} frames). "
                             f"Please upload a clip of a single sign, up to {int(max_duration_s)} seconds.")
    cap.release()
    if total < MIN_FRAMES:
        raise VideoError("No readable frames were found -- the file may be corrupted or use an unsupported codec.")
    duration = total / fps
    if duration > max_duration_s:
        raise VideoError(f"The video is {duration:.1f} s long. Please upload a clip of a single sign, "
                         f"up to {int(max_duration_s)} seconds.")

    indices = sample_frame_indices(total)
    wanted = set(indices)
    kept = {}
    cap = cv2.VideoCapture(str(video_path))
    i = 0
    while i <= max(indices):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            kept[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    if len(kept) < len(wanted):                                # decoder returned fewer frames on 2nd pass
        last = kept[max(kept)] if kept else None
        if last is None:
            raise VideoError("The video frames could not be decoded.")
        for j in wanted - kept.keys():
            kept[j] = last
    info = VideoInfo(fps=float(fps), total_frames=total, width=width, height=height,
                     duration_s=float(duration), sampled_indices=indices)
    return [kept[j] for j in indices], info


# ------------------------------------------------------------------ MediaPipe (verbatim logic)
class _LandmarkListWrapper:
    def __init__(self, landmark_list):
        self.landmark = landmark_list


def to_legacy_style_results(hand_result, pose_result, face_result):
    left_hand_landmarks, right_hand_landmarks = None, None
    if hand_result.hand_landmarks:
        hands_with_x = [(lm_list[0].x, lm_list) for lm_list in hand_result.hand_landmarks]
        hands_with_x.sort(key=lambda h: h[0])
        if len(hands_with_x) >= 1:
            left_hand_landmarks = _LandmarkListWrapper(hands_with_x[0][1])
        if len(hands_with_x) >= 2:
            right_hand_landmarks = _LandmarkListWrapper(hands_with_x[-1][1])
    pose_landmarks = _LandmarkListWrapper(pose_result.pose_landmarks[0]) if pose_result.pose_landmarks else None
    face_landmarks = _LandmarkListWrapper(face_result.face_landmarks[0]) if face_result.face_landmarks else None
    return types.SimpleNamespace(left_hand_landmarks=left_hand_landmarks, right_hand_landmarks=right_hand_landmarks,
                                 pose_landmarks=pose_landmarks, face_landmarks=face_landmarks)


def get_signer_bbox(pose_landmarks, frame_shape, margin=SIGNER_BBOX_MARGIN):
    h, w = frame_shape[:2]
    if pose_landmarks is None:
        return (0, 0, w, h)
    xs = [lm.x for lm in pose_landmarks.landmark]
    ys = [lm.y for lm in pose_landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_w, box_h = x_max - x_min, y_max - y_min
    x_min = max(0.0, x_min - box_w * margin)
    x_max = min(1.0, x_max + box_w * margin)
    y_min = max(0.0, y_min - box_h * margin)
    y_max = min(1.0, y_max + box_h * margin)
    return (int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h))


def crop_signer(frame, bbox, out_size=IMG_SIZE):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return cv2.resize(frame, (out_size, out_size))
    return cv2.resize(frame[y1:y2, x1:x2], (out_size, out_size))


def landmarks_to_vector(results):
    def _xy(lm_list, idx_list=None):
        if lm_list is None:
            n = len(idx_list) if idx_list else 0
            return np.zeros((n, 2), dtype=np.float32)
        pts = lm_list.landmark
        if idx_list is not None:
            pts = [pts[i] for i in idx_list]
        return np.array([[p.x, p.y] for p in pts], dtype=np.float32)

    left_hand = _xy(results.left_hand_landmarks, list(range(N_HAND_LANDMARKS)))
    right_hand = _xy(results.right_hand_landmarks, list(range(N_HAND_LANDMARKS)))
    pose = _xy(results.pose_landmarks, POSE_SUBSET_IDX)
    face = _xy(results.face_landmarks, FACE_SUBSET_IDX)
    if results.pose_landmarks is not None:
        l_sh = results.pose_landmarks.landmark[11]
        r_sh = results.pose_landmarks.landmark[12]
        center = np.array([(l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2])
        scale = max(np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y])), 1e-6)
    else:
        center, scale = np.array([0.5, 0.5]), 1.0
    all_pts = np.concatenate([left_hand, right_hand, pose, face], axis=0)
    return ((all_pts - center) / scale).flatten()


class FeatureExtractor:
    """Holds one set of MediaPipe landmarkers, like the cache builder did (one instance reused across
    videos with a monotonically increasing timestamp counter). A lock keeps it safe under concurrent
    Streamlit sessions."""

    def __init__(self, model_dir=MEDIAPIPE_DIR, reset_each_video=True):
        """reset_each_video=True starts every video with fresh tracking state, so the same upload
        always yields the same features (VIDEO-mode trackers otherwise carry state from the previous
        clip -- see docs/testing_report.md, parity section)."""
        import mediapipe as mp
        self._mp = mp
        self._model_dir = Path(model_dir)
        self.reset_each_video = reset_each_video
        self._lock = threading.Lock()
        self.hand = self.pose = self.face = None
        self._create()

    def _create(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        base = mp_python.BaseOptions
        d = self._model_dir
        self.close()
        self.hand = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(d / "hand_landmarker.task")),
            running_mode=mp_vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.5))
        self.pose = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(d / "pose_landmarker_lite.task")),
            running_mode=mp_vision.RunningMode.VIDEO))
        self.face = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
            base_options=base(model_asset_path=str(d / "face_landmarker.task")),
            running_mode=mp_vision.RunningMode.VIDEO, num_faces=1))
        self._ts = 0
        self._fresh = True

    def extract_from_frames(self, frames, info):
        appearance = np.zeros((FRAME_COUNT, 1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        landmarks = np.zeros((FRAME_COUNT, LANDMARK_VECTOR_DIM), dtype=np.float32)
        n_hands = n_pose = n_face = 0
        with self._lock:
            if self.reset_each_video and not self._fresh:
                self._create()
            self._fresh = False
            for t, frame in enumerate(frames):
                frame = np.ascontiguousarray(frame)
                self._ts += 1
                img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
                res = to_legacy_style_results(self.hand.detect_for_video(img, self._ts),
                                              self.pose.detect_for_video(img, self._ts),
                                              self.face.detect_for_video(img, self._ts))
                appearance[t, 0] = crop_signer(frame, get_signer_bbox(res.pose_landmarks, frame.shape))
                landmarks[t] = landmarks_to_vector(res)
                n_hands += res.left_hand_landmarks is not None or res.right_hand_landmarks is not None
                n_pose += res.pose_landmarks is not None
                n_face += res.face_landmarks is not None
        return ClipFeatures(landmarks=landmarks, appearance=appearance, info=info,
                            frames_with_hands=n_hands, frames_with_pose=n_pose, frames_with_face=n_face)

    def extract(self, video_path):
        frames, info = read_sampled_frames(video_path)
        return self.extract_from_frames(frames, info)

    def close(self):
        for lm in (self.hand, self.pose, self.face):
            if lm is not None:
                lm.close()
        self.hand = self.pose = self.face = None
