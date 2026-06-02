import numpy as np
import threading
import platform
import queue
import subprocess
import os
import cv2
import pyttsx3

# ── Landmark index constants ──────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Iris landmarks (only available when refine_landmarks=True)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOT = 78, 308, 13, 14
NOSE, CHIN, FOREHEAD = 1, 152, 10

# 3-D model points for solvePnP (generic neutral face, millimetres)
_MODEL_3D = np.array([
    [0.0,    0.0,    0.0],    # NOSE tip      (lm 1)
    [0.0,  -330.0, -65.0],    # CHIN          (lm 152)
    [-225.0, 170.0,-135.0],   # LEFT eye corner  (lm 33)
    [ 225.0, 170.0,-135.0],   # RIGHT eye corner (lm 263)
    [-150.0,-150.0,-125.0],   # MOUTH left    (lm 78)
    [ 150.0,-150.0,-125.0],   # MOUTH right   (lm 308)
], dtype=np.float64)

_SOLVEPNP_INDICES = [1, 152, 33, 263, 78, 308]

# ── Singleton TTS engine ──────────────────────────────────────────────────────
class TTSEngine:
    """Thread-safe singleton TTS engine backed by a persistent worker thread.

    Fixes the original bug where pyttsx3.init() was called inside every
    thread, causing COM conflicts on Windows and audio leaks on Linux/macOS.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_engine()
        return cls._instance

    def _init_engine(self):
        self._q = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)
        except Exception as e:
            print(f"[TTS] Engine init failed: {e}")
            engine = None

        while True:
            text = self._q.get()
            if text is None:          # sentinel → shutdown
                break
            if engine is None:
                print(f"[TTS] (no engine) would say: {text}")
                continue
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Error speaking '{text}': {e}")
                # Attempt re-init on next call
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 165)
                except Exception:
                    engine = None

    def speak(self, text: str):
        """Queue a string for speech. Non-blocking."""
        self._q.put(text)

    def shutdown(self):
        self._q.put(None)


# Module-level singleton — import and call tts_engine.speak() anywhere
tts_engine = TTSEngine()


# Keep async_voice as a thin wrapper so existing call-sites still work
def async_voice(text: str):
    tts_engine.speak(text)


# ── Cross-platform beep ───────────────────────────────────────────────────────
def _do_beep():
    sys = platform.system()
    if sys == "Windows":
        import winsound
        winsound.Beep(1000, 400)
    elif sys == "Darwin":
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Ping.aiff"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:  # Linux
        # Try PulseAudio first, fall back to 'beep' command, then terminal bell
        sound = "/usr/share/sounds/freedesktop/stereo/bell.oga"
        if os.path.exists(sound):
            result = subprocess.run(
                ["paplay", sound],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            result = subprocess.run(
                ["beep"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        if result.returncode != 0:
            print("\a", end="", flush=True)


def async_beep():
    threading.Thread(target=_do_beep, daemon=True).start()


# ── Geometry helpers ──────────────────────────────────────────────────────────
def get_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def extract_pts(landmarks, indices, w, h):
    return [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]


def compute_ear(eye_indices, landmarks, w, h):
    p = extract_pts(landmarks, eye_indices, w, h)
    h_dist = get_distance(p[0], p[3])
    if h_dist == 0:
        return 0.0
    return (get_distance(p[1], p[5]) + get_distance(p[2], p[4])) / (2.0 * h_dist)


# ── Head pose via solvePnP ────────────────────────────────────────────────────
def compute_head_pitch(landmarks, img_w, img_h):
    """Return pitch angle in degrees using cv2.solvePnP.

    Positive = head nodding down (drowsy direction).
    Falls back to the old ratio-based estimate if solvePnP fails.
    """
    image_pts = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h]
         for i in _SOLVEPNP_INDICES],
        dtype=np.float64
    )

    focal = float(img_w)
    cx, cy = img_w / 2.0, img_h / 2.0
    camera_matrix = np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rvec, _ = cv2.solvePnP(
            _MODEL_3D, image_pts, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            raise ValueError("solvePnP returned False")
        rmat, _ = cv2.Rodrigues(rvec)
        # pitch: rotation around X axis — negative rmat[2][1] → nod-down is positive
        pitch_deg = float(np.degrees(np.arcsin(-rmat[2][1])))
        return pitch_deg
    except Exception:
        # Fallback: original ratio method
        face_pts = extract_pts(landmarks, [FOREHEAD, CHIN, NOSE], img_w, img_h)
        nose_chin = get_distance(face_pts[2], face_pts[1])
        ratio = get_distance(face_pts[0], face_pts[2]) / nose_chin if nose_chin > 0 else 1.0
        return max(0.0, (ratio - 1.0) * 40.0)


# ── Gaze estimation ───────────────────────────────────────────────────────────
def compute_gaze_offset(landmarks, img_w, img_h):
    """Return (gaze_x, gaze_y) where 0,0 = centre, ±1 = edge.

    Uses MediaPipe iris landmarks (requires refine_landmarks=True).
    Positive x = looking right, positive y = looking down.
    Returns (0.0, 0.0) if iris landmarks are absent.
    """
    def _eye_gaze(iris_indices, corner_indices):
        iris_pts = extract_pts(landmarks, iris_indices, img_w, img_h)
        corner_pts = extract_pts(landmarks, corner_indices, img_w, img_h)

        iris_cx = np.mean([p[0] for p in iris_pts])
        iris_cy = np.mean([p[1] for p in iris_pts])

        eye_left   = corner_pts[0]  # leftmost corner
        eye_right  = corner_pts[3]  # rightmost corner
        eye_top    = corner_pts[1]
        eye_bot    = corner_pts[5]

        eye_mid_x = (eye_left[0] + eye_right[0]) / 2.0
        eye_mid_y = (eye_top[1]  + eye_bot[1])   / 2.0
        eye_w     = max(get_distance(eye_left, eye_right), 1.0)
        eye_h     = max(get_distance(eye_top,  eye_bot),  1.0)

        off_x = (iris_cx - eye_mid_x) / eye_w
        off_y = (iris_cy - eye_mid_y) / eye_h
        return off_x, off_y

    try:
        lx, ly = _eye_gaze(LEFT_IRIS,  LEFT_EYE)
        rx, ry = _eye_gaze(RIGHT_IRIS, RIGHT_EYE)
        return (lx + rx) / 2.0, (ly + ry) / 2.0
    except (IndexError, AttributeError):
        return 0.0, 0.0


# ── Main feature extraction ───────────────────────────────────────────────────
def get_features(landmarks, img_w, img_h):
    """Return (ear, mar, ratio, head_tilt_deg).

    head_tilt_deg is now a true pitch angle from solvePnP.
    ratio is kept for backward ML-model compatibility.
    """
    ear = (
        compute_ear(LEFT_EYE,  landmarks, img_w, img_h) +
        compute_ear(RIGHT_EYE, landmarks, img_w, img_h)
    ) / 2.0

    mouth_pts = extract_pts(
        landmarks, [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOT], img_w, img_h
    )
    h_dist = get_distance(mouth_pts[0], mouth_pts[1])
    mar = get_distance(mouth_pts[2], mouth_pts[3]) / h_dist if h_dist > 0 else 0.0

    # Keep old ratio for ML model feature parity
    face_pts = extract_pts(landmarks, [FOREHEAD, CHIN, NOSE], img_w, img_h)
    nose_chin = get_distance(face_pts[2], face_pts[1])
    ratio = get_distance(face_pts[0], face_pts[2]) / nose_chin if nose_chin > 0 else 1.0

    head_tilt_deg = compute_head_pitch(landmarks, img_w, img_h)

    return ear, mar, ratio, head_tilt_deg
