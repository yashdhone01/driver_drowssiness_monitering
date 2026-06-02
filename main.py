import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from collections import deque
import joblib
import pandas as pd
from datetime import datetime
import json
import psutil

from simulation import DrivingSimulator
from utils import (
    get_features, compute_gaze_offset,
    async_beep, async_voice, tts_engine,
    NOSE
)

# ── Config ────────────────────────────────────────────────────────────────────
try:
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    print("[✔] Loaded config.json successfully.")
except FileNotFoundError:
    print("[!] Warning: config.json not found! Using defaults.")
    CONFIG = {
        "thresholds": {
            "ear": 0.22, "mar": 0.5,
            "head_tilt": 1.6,
            "movement_high": 12.0, "movement_low": 3.0
        },
        "timings": {
            "unconscious_time": 5.0,
            "state_persist_frames": 5,
            "alert_cooldown": 5.0
        },
        "weights": {"eye_closed": 1.5, "yawn": 2.0, "head_tilt": 0.5}
    }

TH = CONFIG["thresholds"]
TM = CONFIG["timings"]
WT = CONFIG["weights"]

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils


# ── KPI Module ────────────────────────────────────────────────────────────────
class KPIEvaluationModule:
    def __init__(self):
        self.total_drowsy_triggers  = 0
        self.false_alarms           = 0
        self.drowsy_start_timestamp = None
        self.mttd_list              = []
        self.fps_estimate           = 30
        self.perclos_buffer         = deque(maxlen=60 * self.fps_estimate)
        self.blink_count            = 0
        self.last_ear               = 1.0
        self.head_pitch_buffer      = deque(maxlen=30)
        self.head_stability_score   = 0.0
        self.fps                    = 0.0
        self.frame_count            = 0
        self.session_start          = time.time()
        self.latency_ms             = 0.0
        self.cpu_usage              = 0.0
        self.ram_usage              = 0.0
        self.last_resource_poll     = 0.0
        self._frame_start_t         = time.perf_counter()
        self.face_lost_time         = None
        self.reacquisition_times    = []

    def process_frame_start(self):
        self._frame_start_t = time.perf_counter()

    def process_frame_end(self):
        self.latency_ms = (time.perf_counter() - self._frame_start_t) * 1000.0
        self.frame_count += 1
        elapsed = time.time() - self.session_start
        if elapsed > 0:
            self.fps = self.frame_count / elapsed

    def async_poll_sys(self):
        now = time.time()
        if now - self.last_resource_poll > 2.0:
            self.cpu_usage          = psutil.cpu_percent(interval=None)
            self.ram_usage          = psutil.virtual_memory().percent
            self.last_resource_poll = now

    def update(self, state, prev_state, current_time, ear,
               is_face_detected, head_tilt_deg, dynamic_ear_th, eyes_closed_start):
        self.async_poll_sys()

        if not is_face_detected:
            if self.face_lost_time is None:
                self.face_lost_time = current_time
        else:
            if self.face_lost_time is not None:
                self.reacquisition_times.append(current_time - self.face_lost_time)
                self.face_lost_time = None

        if is_face_detected:
            self.perclos_buffer.append(1 if ear < dynamic_ear_th else 0)
            if self.last_ear >= dynamic_ear_th and ear < dynamic_ear_th:
                self.blink_count += 1
            self.last_ear = ear

        self.head_pitch_buffer.append(head_tilt_deg)
        if len(self.head_pitch_buffer) >= 3:
            self.head_stability_score = float(np.var(self.head_pitch_buffer))

        if state == "DROWSY" and prev_state != "DROWSY":
            self.total_drowsy_triggers += 1
            self.drowsy_start_timestamp = current_time
            if eyes_closed_start:
                self.mttd_list.append(current_time - eyes_closed_start)

        if prev_state == "DROWSY" and state == "ALERT":
            if self.drowsy_start_timestamp and (current_time - self.drowsy_start_timestamp) < 1.5:
                self.false_alarms += 1

    def output_kpi_summary(self):
        elapsed_min   = (time.time() - self.session_start) / 60.0
        safe_min      = elapsed_min if elapsed_min > 0 else 1.0
        avg_mttd      = float(np.mean(self.mttd_list)) if self.mttd_list else 0.0
        avg_reacq     = float(np.mean(self.reacquisition_times)) if self.reacquisition_times else 0.0
        perclos_pct   = (sum(self.perclos_buffer) / len(self.perclos_buffer) * 100) if self.perclos_buffer else 0.0
        blink_rate    = self.blink_count / safe_min

        print("\n" + "=" * 50)
        print(" 🛡 ROAD-READY SYSTEM KPI SUMMARY")
        print("=" * 50)
        print(f"  FDR (False Alarms):  {self.false_alarms}")
        print(f"  MTTD (Avg Delay):    {avg_mttd:.2f}s")
        print(f"  PERCLOS:             {perclos_pct:.1f}%")
        print(f"  Avg Blink Rate:      {blink_rate:.1f} BPM")
        print(f"  FPS:                 {self.fps:.1f}")
        print(f"  Latency:             {self.latency_ms:.1f} ms")
        print(f"  Face Reacquisition:  {avg_reacq:.2f}s avg")
        print("=" * 50 + "\n")


# ── Session Analytics ─────────────────────────────────────────────────────────
class SessionAnalytics:
    def __init__(self):
        self.start_time        = time.time()
        self.total_alert       = 0.0
        self.total_drowsy      = 0.0
        self.total_unconscious = 0.0
        self.total_distracted  = 0.0
        self.num_alerts        = 0
        self.max_fatigue       = 0.0
        self.last_update       = time.time()

    def update(self, current_state, fatigue_score):
        now = time.time()
        dt  = now - self.last_update
        if   current_state == "ALERT":       self.total_alert       += dt
        elif current_state == "DROWSY":      self.total_drowsy      += dt
        elif current_state == "UNCONSCIOUS": self.total_unconscious += dt
        elif current_state == "DISTRACTED":  self.total_distracted  += dt
        if fatigue_score > self.max_fatigue:
            self.max_fatigue = fatigue_score
        self.last_update = now

    def print_summary(self):
        total = time.time() - self.start_time
        s     = total if total > 0 else 1
        print("\n" + "=" * 50)
        print(" 📊 SESSION ANALYTICS")
        print("=" * 50)
        print(f"  Total Time:   {total:.1f}s")
        print(f"  ALERT:        {self.total_alert:.1f}s  ({self.total_alert/s*100:.1f}%)")
        print(f"  DROWSY:       {self.total_drowsy:.1f}s  ({self.total_drowsy/s*100:.1f}%)")
        print(f"  DISTRACTED:   {self.total_distracted:.1f}s  ({self.total_distracted/s*100:.1f}%)")
        print(f"  UNCONSCIOUS:  {self.total_unconscious:.1f}s  ({self.total_unconscious/s*100:.1f}%)")
        print(f"  Alerts:       {self.num_alerts}")
        print(f"  Peak Fatigue: {self.max_fatigue:.1f}")
        print("=" * 50)


# ── Main DMS class ────────────────────────────────────────────────────────────
class HybridDriverSystem:
    def __init__(self):
        # ── Core state ──
        self.final_state = "ALERT"
        self.prev_state  = "ALERT"
        self.analytics   = SessionAnalytics()
        self.kpi         = KPIEvaluationModule()

        # ── ML model ──
        try:
            self.model = joblib.load("model.pkl")
            print("[✔] ML Model loaded: 'model.pkl'")
        except FileNotFoundError:
            print("[X] model.pkl not found — rule-based fallback active.")
            self.model = None

        # ── Timing trackers (all declared here — no more getattr fallbacks) ──
        self.eyes_closed_start   = None
        self.eyes_open_start     = None
        self.head_down_start     = None
        self.min_movement_start  = None
        self.microsleep_start    = None
        self.high_fatigue_start  = None
        self.drowsy_start_time   = None
        self.unconscious_start_time = None
        self.emergency_start_time   = None

        # ── Audio one-shot flags (all declared here) ──
        self.d1_played            = False
        self.d2_played            = False
        self.u1_played            = False
        self.u2_played            = False
        self.u3_played            = False
        self.emergency_audio_played = False
        self.emergency_alert_sent   = False

        # ── Fatigue / scoring ──
        self.fatigue_score   = 0.0
        self.max_fatigue     = 100.0
        self.prev_head_tilt  = 0.0
        self.confidence      = 100.0
        self.last_ml_state        = "ALERT"
        self.ml_state_change_time = 0.0

        # ── Escalation stage counters ──
        self.drowsy_stage = 0
        self.unc_stage    = 0

        # ── Buffers ──
        self.nose_history    = deque(maxlen=5)
        self.state_history   = deque(maxlen=20)
        self.raw_ear_buffer  = deque(maxlen=5)
        self.drowsy_timestamps = []
        self.timeline        = deque(maxlen=30)
        self.perclos_window  = deque(maxlen=int(30 * 30))   # 30s @ ~30fps
        self.perclos         = 0.0
        self.microsleep_count = 0

        # ── Movement / gaze ──
        self.movement_status = "MEDIUM"
        self.eye_time        = 0.0
        self.no_move_time    = 0.0
        self.gaze_x          = 0.0
        self.gaze_y          = 0.0
        self.gaze_distracted_start = None

        # ── Raw readouts for HUD transparency ──
        self.last_raw_ear  = 0.0
        self.last_raw_mar  = 0.0
        self.last_raw_tilt = 0.0

        # ── Misc ──
        self.last_audio_time       = 0.0
        self.last_alert_time       = 0.0
        self.take_trigger_screenshot = False
        self.debug_override        = None
        self.show_kpi_hud          = False
        self.is_night_vision       = False
        self.unconscious_streak    = 0
        self.api_notification_cooldown = 0

        # ── Calibration ──
        self.calibrating           = True
        self.calibration_start_time = None
        self.calibration_duration  = 5.0
        self.calib_ear  = []
        self.calib_mar  = []
        self.calib_head = []
        self.dynamic_ear_th  = TH["ear"]
        self.dynamic_mar_th  = TH["mar"]
        self.dynamic_head_th = TH["head_tilt"]

    # ── Helper: reset drowsy audio flags ─────────────────────────────────────
    def _reset_drowsy_flags(self):
        self.d1_played       = False
        self.d2_played       = False
        self.drowsy_start_time = None
        self.drowsy_stage    = 0

    # ── Helper: reset unconscious audio flags ─────────────────────────────────
    def _reset_unconscious_flags(self):
        self.u1_played              = False
        self.u2_played              = False
        self.u3_played              = False
        self.unconscious_start_time = None
        self.unc_stage              = 0

    # ── Helper: reset emergency flags ────────────────────────────────────────
    def _reset_emergency_flags(self):
        self.emergency_audio_played = False
        self.emergency_alert_sent   = False
        self.emergency_start_time   = None

    def log_event(self, extra_type=None):
        import os
        os.makedirs("incidents", exist_ok=True)
        csv_file     = "incidents/events_log.csv"
        write_header = not os.path.exists(csv_file)
        ts_str       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event_type   = extra_type if extra_type else self.final_state
        with open(csv_file, "a") as f:
            if write_header:
                f.write("timestamp,event_type,EAR,MAR,head_tilt,fatigue,perclos\n")
            f.write(
                f"{ts_str},{event_type},{self.last_raw_ear:.3f},"
                f"{self.last_raw_mar:.3f},{self.last_raw_tilt:.1f},"
                f"{self.fatigue_score:.1f},{self.perclos:.3f}\n"
            )

    def process_logic(self, raw_ear, mar, ratio, head_tilt_deg,
                      nose_pos, current_time, is_face_detected,
                      gaze_offset=(0.0, 0.0)):
        self.kpi.process_frame_start()

        self.last_raw_ear  = raw_ear
        self.last_raw_mar  = mar
        self.last_raw_tilt = head_tilt_deg
        self.gaze_x, self.gaze_y = gaze_offset

        if not is_face_detected:
            self.eyes_closed_start  = None
            self.head_down_start    = None
            self.microsleep_start   = None
            self.kpi.update(self.final_state, self.prev_state, current_time,
                            raw_ear, False, 0.0, self.dynamic_ear_th, None)
            self.kpi.process_frame_end()
            return

        # ── Prune old drowsy timestamps ──
        self.drowsy_timestamps = [
            t for t in self.drowsy_timestamps if current_time - t <= 100.0
        ]

        # ── EAR smoothing ──
        self.raw_ear_buffer.append(raw_ear)
        ear = float(np.mean(self.raw_ear_buffer))

        # ── Calibration phase ──
        if self.calibrating:
            if not self.calibration_start_time:
                self.calibration_start_time = current_time
            if current_time - self.calibration_start_time < self.calibration_duration:
                self.calib_ear.append(ear)
                self.calib_mar.append(mar)
                self.calib_head.append(ratio)
                self.final_state = "ALERT"
                self.nose_history.append(nose_pos)
                self.kpi.update(self.final_state, self.prev_state, current_time,
                                ear, True, head_tilt_deg, self.dynamic_ear_th, None)
                self.kpi.process_frame_end()
                return
            else:
                self.calibrating = False
                if len(self.calib_ear) > 15:
                    baseline_ear  = float(np.mean(self.calib_ear))
                    baseline_mar  = float(np.mean(self.calib_mar))
                    baseline_head = float(np.mean(self.calib_head))
                    self.dynamic_ear_th  = max(0.12, baseline_ear - 0.08)
                    self.dynamic_mar_th  = baseline_mar + 0.3
                    self.dynamic_head_th = baseline_head + 0.4
                    CONFIG["last_baseline_ear"] = baseline_ear
                    with open('config.json', 'w') as f:
                        json.dump(CONFIG, f, indent=4)
                    print(f"[✔] Calibration done — Baseline EAR={baseline_ear:.2f}, "
                          f"Threshold={self.dynamic_ear_th:.2f}")
                else:
                    print("[!] Too few calibration frames — using defaults.")

        # ── PERCLOS rolling window ──
        self.perclos_window.append(1 if ear < self.dynamic_ear_th else 0)
        self.perclos = sum(self.perclos_window) / len(self.perclos_window)

        # ── Gaze distraction detection ──
        if abs(self.gaze_x) > 0.35:
            if self.gaze_distracted_start is None:
                self.gaze_distracted_start = current_time
        else:
            self.gaze_distracted_start = None

        gaze_distracted = (
            self.gaze_distracted_start is not None and
            (current_time - self.gaze_distracted_start) > 3.0
        )

        # ── Eye open/closed trackers ──
        margin = (self.dynamic_ear_th + 0.02
                  if "DROWSY" in self.final_state or "UNCONSCIOUS" in self.final_state
                  else self.dynamic_ear_th)

        if ear < margin:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            self.eyes_open_start = None
        else:
            self.eyes_closed_start = None
            if self.eyes_open_start is None:
                self.eyes_open_start = current_time

        self.eye_time       = (current_time - self.eyes_closed_start
                               if self.eyes_closed_start else 0.0)
        time_eyes_open      = (current_time - self.eyes_open_start
                               if self.eyes_open_start else 0.0)
        eyes_open_consistently = time_eyes_open > 2.0

        # ── Microsleep detection ──
        if ear < (self.dynamic_ear_th - 0.05):
            if self.microsleep_start is None:
                self.microsleep_start = current_time
        else:
            if self.microsleep_start is not None:
                ms_dur = current_time - self.microsleep_start
                if ms_dur >= 0.5:   # genuine microsleep event
                    self.microsleep_count += 1
                    self.log_event(extra_type=f"MICROSLEEP_{ms_dur:.1f}s")
            self.microsleep_start = None

        microsleep_dur = (current_time - self.microsleep_start
                          if self.microsleep_start else 0.0)

        # ── Head down tracker ──
        head_down = ratio > self.dynamic_head_th
        if head_down:
            if self.head_down_start is None:
                self.head_down_start = current_time
        else:
            self.head_down_start = None
        head_time = current_time - self.head_down_start if self.head_down_start else 0.0

        # ── Movement tracker ──
        self.nose_history.append(nose_pos)
        if len(self.nose_history) >= 3:
            xs  = [p[0] for p in self.nose_history]
            ys  = [p[1] for p in self.nose_history]
            var = np.var(xs) + np.var(ys)
            if   var > TH["movement_high"]: self.movement_status = "HIGH"
            elif var < TH["movement_low"]:  self.movement_status = "LOW"
            else:                           self.movement_status = "MEDIUM"

        minimal_movement = self.movement_status == "LOW"
        if minimal_movement:
            if self.min_movement_start is None:
                self.min_movement_start = current_time
        else:
            self.min_movement_start = None
        self.no_move_time = (current_time - self.min_movement_start
                             if self.min_movement_start else 0.0)

        # ── ML prediction ──
        if self.model:
            features_df = pd.DataFrame(
                [[ear, mar, head_tilt_deg, self.eye_time, self.no_move_time]],
                columns=['EAR', 'mouth_ratio', 'head_tilt',
                         'eye_closure_time', 'no_movement_time']
            )
            ml_pred     = self.model.predict(features_df)[0]
            probs       = self.model.predict_proba(features_df)[0]
            class_idx   = list(self.model.classes_).index(ml_pred)
            self.confidence = float(probs[class_idx]) * 100.0
            raw_ml = {"unconscious": "UNCONSCIOUS",
                      "drowsy":      "DROWSY"}.get(ml_pred, "ALERT")
        else:
            raw_ml = "ALERT"

        # ── Fatigue accumulator ──
        self.fatigue_score -= 0.1   # passive leak

        if raw_ml != self.last_ml_state:
            self.last_ml_state        = raw_ml
            self.ml_state_change_time = current_time

        in_grace = (current_time - self.ml_state_change_time) < 2.0

        if not in_grace:
            if   self.fatigue_score < 30.0: multiplier = 0.5
            elif self.fatigue_score < 70.0: multiplier = 1.0
            else:                           multiplier = 1.5

            if raw_ml == "UNCONSCIOUS":
                self.fatigue_score += 1.5 * multiplier
            elif raw_ml == "DROWSY":
                self.fatigue_score += 0.8 * multiplier

            # Microsleep bonus (heavier than standard low EAR)
            if microsleep_dur >= 0.5:
                self.fatigue_score += 3.0 * multiplier
            elif ear < self.dynamic_ear_th - 0.05:
                self.fatigue_score += 2.0 * multiplier

            if mar > self.dynamic_mar_th:
                self.fatigue_score += 0.5 * multiplier

            # PERCLOS sustained bonus
            if self.perclos > 0.25:
                self.fatigue_score += 0.4 * multiplier

            # Recovery
            delta_tilt   = abs(head_tilt_deg - self.prev_head_tilt)
            movement_high = delta_tilt > 5.0
            if movement_high or eyes_open_consistently:
                self.fatigue_score -= 8.0

        self.prev_head_tilt  = head_tilt_deg
        self.fatigue_score   = max(0.0, min(self.fatigue_score, 100.0))

        # ── Sustained high-fatigue gate ──
        if self.fatigue_score > 85.0:
            if self.high_fatigue_start is None:
                self.high_fatigue_start = current_time
        else:
            self.high_fatigue_start = None

        sustained_condition = (
            self.high_fatigue_start is not None and
            (current_time - self.high_fatigue_start) > 2.5
        )

        # ── Override condition (catastrophic instant escalation) ──
        hard_eye_closure  = ear < max(0.12, self.dynamic_ear_th - 0.04)
        override_condition = (
            self.eye_time > 2.0 and
            self.movement_status == "LOW" and
            hard_eye_closure and
            head_down
        )
        if override_condition:
            self.fatigue_score     = 100.0
            if self.high_fatigue_start is None:
                self.high_fatigue_start = current_time - 3.0

        # ── FSM decision ──
        if (self.final_state == "EMERGENCY" and
                self.emergency_start_time is not None and
                (current_time - self.emergency_start_time) < 5.0):
            self.final_state = "EMERGENCY"           # minimum 5s lock
        elif override_condition:
            self.final_state = "UNCONSCIOUS"
        elif self.fatigue_score > 85.0 and sustained_condition:
            self.final_state = "UNCONSCIOUS"
        elif self.fatigue_score > 50.0:
            self.final_state = "DROWSY"
        elif gaze_distracted:
            self.final_state = "DISTRACTED"
        else:
            self.final_state = "ALERT"

        # ── Debug override injection ──
        if self.debug_override:
            self.final_state = self.debug_override
            if self.debug_override == "UNCONSCIOUS":
                self.fatigue_score = 100.0
            elif self.debug_override == "DROWSY":
                self.fatigue_score = 60.0

        # ── Debug print every 10 frames ──
        if self.kpi.frame_count % 10 == 0:
            print(f"Override:{override_condition} | "
                  f"EAR_dur:{self.eye_time:.1f}s | "
                  f"Mov:{self.movement_status} | "
                  f"Score:{self.fatigue_score:.1f} | "
                  f"PERCLOS:{self.perclos*100:.1f}% | "
                  f"Gaze:{self.gaze_x:+.2f} | "
                  f"State:{self.final_state}")

        # ── Multi-stage escalation action layer ──
        audio_ok = (current_time - self.last_audio_time) > 2.0

        if self.final_state in ("ALERT", "DISTRACTED"):
            self.unconscious_start_time = None
            self._reset_drowsy_flags()
            self._reset_unconscious_flags()
            self._reset_emergency_flags()

        elif self.final_state == "DROWSY":
            self._reset_unconscious_flags()
            self._reset_emergency_flags()

            if self.drowsy_start_time is None:
                self.drowsy_start_time = current_time
                self.drowsy_stage      = 1

            drowsy_elapsed = current_time - self.drowsy_start_time

            if drowsy_elapsed < 10.0:
                self.drowsy_stage = 1
            elif drowsy_elapsed < 30.0:
                self.drowsy_stage = 2
                if not self.d1_played and audio_ok:
                    async_beep()
                    async_voice("Consider a coffee break")
                    self.last_audio_time = current_time
                    self.d1_played = True
            else:
                self.drowsy_stage = 3
                if not self.d2_played and audio_ok:
                    async_beep()
                    async_voice("Please take a halt")
                    self.last_audio_time = current_time
                    self.d2_played = True

        elif self.final_state == "UNCONSCIOUS":
            self._reset_drowsy_flags()
            self._reset_emergency_flags()

            if self.unconscious_start_time is None:
                self.unconscious_start_time = current_time
                self.unc_stage              = 1

            unc_elapsed = current_time - self.unconscious_start_time

            if unc_elapsed < 3.0:
                self.unc_stage = 1    # silent visual-only window

            elif unc_elapsed < 15.0:
                self.unc_stage = 1
                if not self.u1_played and audio_ok:
                    async_beep()
                    async_voice("Are you awake?")
                    self.last_audio_time = current_time
                    self.u1_played = True

            elif unc_elapsed < 25.0:
                self.unc_stage = 2
                if not self.u2_played and audio_ok:
                    async_beep()
                    async_voice("Please hold the steering wheel. Wake up!")
                    self.last_audio_time = current_time
                    self.u2_played = True

            elif unc_elapsed < 30.0:
                self.unc_stage = 3
                if not self.u3_played and audio_ok:
                    # Cross-platform alarm — no winsound import at call-site
                    async_beep()
                    async_beep()
                    self.last_audio_time = current_time
                    self.u3_played = True
            else:
                # Transition to EMERGENCY
                self.final_state          = "EMERGENCY"
                self.emergency_start_time = current_time

        elif self.final_state == "EMERGENCY":
            if self.emergency_start_time is None:
                self.emergency_start_time = current_time

            if not self.emergency_audio_played:
                def _emergency_audio():
                    async_voice("Calling emergency services")
                    time.sleep(2.5)
                    async_voice("Help is on the way")
                threading.Thread(target=_emergency_audio, daemon=True).start()
                self.emergency_audio_played = True

            if not self.emergency_alert_sent:
                print("\n[!!! DANGER !!!] SOS TRIGGERED — placeholder for Telegram/Email API.")
                self.emergency_alert_sent = True
                # Cross-platform alarm beeps — no winsound here
                for _ in range(3):
                    async_beep()

        # ── State-change side effects ──
        if self.final_state != self.prev_state:
            if self.final_state in ("DROWSY", "UNCONSCIOUS"):
                if self.final_state == "DROWSY":
                    self.drowsy_timestamps.append(current_time)
                if current_time - self.last_alert_time > TM["alert_cooldown"]:
                    self.take_trigger_screenshot = True
                    self.analytics.num_alerts   += 1
                    self.log_event()
                    self.last_alert_time = current_time

        self.analytics.update(self.final_state, self.fatigue_score)
        self.timeline.append(self.final_state)
        self.kpi.update(self.final_state, self.prev_state, current_time, ear,
                        True, head_tilt_deg, self.dynamic_ear_th, self.eyes_closed_start)
        self.prev_state = self.final_state
        self.kpi.process_frame_end()


# ── HUD renderer ──────────────────────────────────────────────────────────────
def draw_hud(frame, monitor, bbox, img_w, img_h, current_time, simulated_dimming):
    ambient     = frame.copy()
    flash_bg    = False
    final_state = monitor.final_state

    if simulated_dimming:
        cv2.putText(frame, "EXPERIMENTAL: LIGHTING SIMULATION (DIMMED)",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    if monitor.is_night_vision:
        cv2.rectangle(frame, (10, img_h - 75), (250, img_h - 60), (0, 0, 0), -1)
        cv2.putText(frame, "[CLAHE Night Vision Active]",
                    (15, img_h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)

    if monitor.calibrating:
        color = (255, 255, 255)
        txt   = "CALIBRATING... PLEASE WAIT"
    elif final_state == "ALERT":
        color = (0, 255, 0)
        txt   = "ALERT (STABLE)"
        cv2.rectangle(ambient, (0, 0), (img_w, img_h), (0, 255, 0), -1)
        frame = cv2.addWeighted(ambient, 0.05, frame, 0.95, 0)
    elif final_state == "DISTRACTED":
        color = (0, 165, 255)
        txt   = "DISTRACTED (EYES OFF ROAD)"
        cv2.rectangle(frame, (0, 0), (img_w, img_h), (0, 165, 255), 6)
    elif final_state == "DROWSY":
        color = (0, 255, 255)
        txt   = "DROWSY (MED RISK)"
        cv2.rectangle(ambient, (0, 0), (img_w, img_h), (0, 255, 255), -1)
        frame = cv2.addWeighted(ambient, 0.2, frame, 0.8, 0)
        d_elapsed = (current_time - monitor.drowsy_start_time
                     if monitor.drowsy_start_time else 0.0)
        cv2.putText(frame, f"[STAGE {monitor.drowsy_stage} - {d_elapsed:.1f}s]",
                    (img_w - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif final_state == "UNCONSCIOUS":
        color    = (0, 0, 255) if int(current_time * 6) % 2 == 0 else (200, 200, 255)
        txt      = "UNCONSCIOUS (!)"
        flash_bg = (color == (0, 0, 255))
        u_elapsed = (current_time - monitor.unconscious_start_time
                     if monitor.unconscious_start_time else 0.0)
        cv2.putText(frame, f"[STAGE {monitor.unc_stage} - {u_elapsed:.1f}s]",
                    (img_w - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if monitor.emergency_alert_sent:
            cv2.putText(frame, "SOS ALERT TRIGGERED",
                        (img_w // 2 - 200, img_h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    elif final_state == "EMERGENCY":
        color    = (0, 0, 255) if int(current_time * 10) % 2 == 0 else (255, 255, 255)
        txt      = "EMERGENCY: SYSTEM LOCKED"
        flash_bg = True
        cv2.putText(frame, "DOORS UNLOCKED. HELP IS ON THE WAY.",
                    (img_w // 2 - 300, img_h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        if monitor.emergency_alert_sent:
            cv2.putText(frame, "SOS ALERT TRIGGERED",
                        (img_w // 2 - 200, img_h // 2 + 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    else:
        color = (255, 255, 255)
        txt   = final_state

    # Border flash
    if flash_bg:
        cv2.rectangle(frame, (0, 0), (img_w, img_h), (0, 0, 255), 10)
        cv2.rectangle(ambient, (0, 0), (img_w, img_h), (0, 0, 255), -1)
        frame = cv2.addWeighted(ambient, 0.35, frame, 0.65, 0)
    elif final_state == "ALERT":
        cv2.rectangle(frame, (0, 0), (img_w, img_h), (0, 255, 0), 6)
    elif final_state in ("DROWSY", "DISTRACTED"):
        cv2.rectangle(frame, (0, 0), (img_w, img_h), (0, 255, 255), 6)

    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # Main info panel
    panel = frame.copy()
    cv2.rectangle(panel, (10, 30), (480, 255), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, 0.7, frame, 0.3, 0)

    cv2.putText(frame, txt, (20, 70),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Fatigue Score: {monitor.fatigue_score:.1f}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"ML Confidence: {monitor.confidence:.1f}%",
                (260, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 1)
    cv2.putText(frame, f"Movement: {monitor.movement_status}",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    calib_str = ("Calibrating..." if monitor.calibrating
                 else f"Baseline EAR: {monitor.dynamic_ear_th + 0.08:.2f}")
    cv2.putText(frame, calib_str, (230, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # Gaze direction label
    gaze_label = ("LEFT" if monitor.gaze_x < -0.2
                  else "RIGHT" if monitor.gaze_x > 0.2 else "CENTER")
    gaze_col   = (0, 165, 255) if gaze_label != "CENTER" else (200, 200, 200)
    cv2.putText(frame, f"GAZE: {gaze_label}  MICROSLEEPS: {monitor.microsleep_count}",
                (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gaze_col, 1)

    # Fatigue bar
    bar_w, bar_h = 400, 20
    px, py = 20, 200
    cv2.putText(frame, "Accumulator", (px, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(frame, (px, py), (px + bar_w, py + bar_h), (50, 50, 50), -1)
    fill_w  = max(0, min(bar_w, int(monitor.fatigue_score / 100.0 * bar_w)))
    bar_col = ((0, 255, 0)   if monitor.fatigue_score < 35
               else (0, 255, 255) if monitor.fatigue_score < 70
               else (0, 0, 255))
    cv2.rectangle(frame, (px, py), (px + fill_w, py + bar_h), bar_col, -1)
    cv2.rectangle(frame, (px, py), (px + bar_w, py + bar_h), (120, 120, 120), 2)
    cv2.putText(frame, f"{monitor.fatigue_score:.1f}/100",
                (px + bar_w + 10, py + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_col, 2)

    # PERCLOS bar (below fatigue bar)
    py2 = py + 30
    cv2.putText(frame, "PERCLOS", (px, py2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.rectangle(frame, (px, py2), (px + bar_w, py2 + 12), (40, 40, 40), -1)
    p_fill = max(0, min(bar_w, int(monitor.perclos * bar_w)))
    p_col  = (0, 255, 0) if monitor.perclos < 0.15 else (0, 165, 255) if monitor.perclos < 0.25 else (0, 0, 255)
    cv2.rectangle(frame, (px, py2), (px + p_fill, py2 + 12), p_col, -1)
    cv2.putText(frame, f"{monitor.perclos*100:.1f}%",
                (px + bar_w + 10, py2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, p_col, 1)

    # Raw signal panel (top right)
    sys_x = img_w - 145
    cv2.rectangle(frame, (sys_x - 10, 20), (img_w - 10, 115), (20, 20, 20), -1)
    cv2.putText(frame, f"EAR:  {monitor.last_raw_ear:.2f}",
                (sys_x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"MAR:  {monitor.last_raw_mar:.2f}",
                (sys_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"TILT: {monitor.last_raw_tilt:.1f}",
                (sys_x, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # KPI overlay (toggle with K)
    if monitor.show_kpi_hud:
        kpi   = monitor.kpi
        koi   = frame.copy()
        pw    = 265
        cv2.rectangle(koi, (img_w - pw - 10, 130), (img_w - 10, 380), (30, 30, 30), -1)
        frame = cv2.addWeighted(koi, 0.85, frame, 0.15, 0)
        px2   = img_w - pw

        perclos_pct = (sum(kpi.perclos_buffer) / len(kpi.perclos_buffer) * 100
                       if kpi.perclos_buffer else 0.0)
        elapsed_min = (time.time() - kpi.session_start) / 60.0
        blink_rate  = kpi.blink_count / max(elapsed_min, 1e-6)
        avg_mttd    = float(np.mean(kpi.mttd_list)) if kpi.mttd_list else 0.0

        for i, (label, val) in enumerate([
            ("[ KPI TELEMETRY ]", None),
            (f"FPS: {kpi.fps:.1f}", (0, 255, 0)),
            (f"Latency: {kpi.latency_ms:.1f} ms", (0, 255, 0)),
            (f"PERCLOS: {perclos_pct:.1f}%", (0, 255, 255)),
            (f"Blink BPM: {blink_rate:.1f}", (0, 255, 255)),
            (f"Head Stab: {kpi.head_stability_score:.2f}", (0, 255, 255)),
            (f"CPU: {kpi.cpu_usage}%", (150, 150, 150)),
            (f"RAM: {kpi.ram_usage}%", (150, 150, 150)),
            (f"MTTD: {avg_mttd:.2f}s", (100, 100, 255)),
            (f"Microsleeps: {monitor.microsleep_count}", (100, 100, 255)),
        ]):
            c = val if val else (255, 255, 255)
            cv2.putText(frame, label, (px2, 155 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 1)

    # Timeline bar
    if monitor.timeline:
        cv2.putText(frame,
                    "Timeline: " + " ".join(s[0] for s in monitor.timeline),
                    (10, img_h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        bw, bh, sx, sy = 12, 10, 10, img_h - 35
        for i, st in enumerate(monitor.timeline):
            c = ((0, 255, 0)   if st == "ALERT"
                 else (0, 255, 255) if st == "DROWSY"
                 else (0, 0, 255)   if st == "UNCONSCIOUS"
                 else (0, 165, 255))
            cv2.rectangle(frame,
                          (sx + i * (bw + 2), sy),
                          (sx + i * (bw + 2) + bw, sy + bh), c, -1)

    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    # Attempt DirectShow first for Windows to avoid MSMF hangs
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    monitor = HybridDriverSystem()
    sim     = DrivingSimulator()

    show_landmarks    = True
    simulate_dimming  = False

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,        # required for iris gaze
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    # Pump events to prevent Pygame window from saying 'Not responding'
                    if sim.should_quit():
                        break
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)

                if simulate_dimming:
                    frame = cv2.convertScaleAbs(frame, alpha=0.35, beta=0)

                # Night-vision CLAHE
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = float(np.mean(gray))
                monitor.is_night_vision = False
                if avg_brightness < 70:
                    monitor.is_night_vision = True
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    cl    = clahe.apply(l)
                    frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

                raw_frame = frame.copy()
                img_h, img_w, _ = frame.shape
                current_time    = time.time()

                frame.flags.writeable = False
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = face_mesh.process(frame_rgb)
                frame.flags.writeable = True

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        lm = face_landmarks.landmark

                        ear, mar, ratio, head_tilt_deg = get_features(
                            lm, img_w, img_h
                        )
                        gaze_offset = compute_gaze_offset(lm, img_w, img_h)
                        nose_pos = (
                            int(lm[NOSE].x * img_w),
                            int(lm[NOSE].y * img_h)
                        )

                        monitor.process_logic(
                            ear, mar, ratio, head_tilt_deg,
                            nose_pos, current_time,
                            is_face_detected=True,
                            gaze_offset=gaze_offset
                        )

                        x_coords = [int(l.x * img_w) for l in lm]
                        y_coords = [int(l.y * img_h) for l in lm]
                        bbox = (
                            min(x_coords), min(y_coords),
                            max(x_coords) - min(x_coords),
                            max(y_coords) - min(y_coords)
                        )

                        if monitor.take_trigger_screenshot:
                            import os
                            os.makedirs("incidents", exist_ok=True)
                            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                            fn  = f"incidents/event_{ts}_{monitor.final_state}.jpg"
                            cv2.imwrite(fn, raw_frame)
                            monitor.take_trigger_screenshot = False

                        frame = draw_hud(
                            frame, monitor, bbox,
                            img_w, img_h, current_time, simulate_dimming
                        )

                        if show_landmarks:
                            mesh_color = {
                                "ALERT":       (255, 0, 0),
                                "UNCONSCIOUS": (0, 0, 255),
                            }.get(monitor.final_state, (0, 165, 255))
                            spec = mp_drawing.DrawingSpec(
                                color=mesh_color, thickness=1, circle_radius=1
                            )
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                connection_drawing_spec=spec
                            )
                else:
                    cv2.putText(frame, "Driver Tracking Lost ⚠",
                                (img_w // 2 - 180, img_h // 2),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                    monitor.process_logic(
                        0.0, 0.0, 0.0, 0.0, (0, 0),
                        current_time, is_face_detected=False
                    )

                cv2.putText(
                    frame,
                    "Debug: [1] Drowsy  [2] Unconscious  [0] ML  "
                    "[L] Mesh  [K] KPI  [D] Dim  [Q] Quit",
                    (10, img_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1
                )
                cv2.imshow('Production Driver Monitor OS', frame)

                # Pygame sync
                drowsy_count = len(monitor.drowsy_timestamps)
                should_hazard = sim.update(monitor.final_state, drowsy_count)
                sim.render(monitor.final_state, should_hazard, drowsy_count)

                if sim.should_quit():
                    break

                key = cv2.waitKey(1) & 0xFF
                active_key = chr(key).lower() if key != 255 else None
                if getattr(sim, 'pressed_key', None):
                    active_key = str(sim.pressed_key).lower()

                if   active_key == 'q': break
                elif active_key == '1': monitor.debug_override = "DROWSY"
                elif active_key == '2': monitor.debug_override = "UNCONSCIOUS"
                elif active_key == '0': monitor.debug_override = None
                elif active_key == 'l': show_landmarks = not show_landmarks
                elif active_key == 'k': monitor.show_kpi_hud = not monitor.show_kpi_hud
                elif active_key == 'd': simulate_dimming = not simulate_dimming

        finally:
            cap.release()
            cv2.destroyAllWindows()
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            tts_engine.shutdown()
            monitor.analytics.print_summary()
            monitor.kpi.output_kpi_summary()


if __name__ == "__main__":
    main()
