# posture_expression.py
import time
import cv2
import mediapipe as mp
from deepface import DeepFace
import pyttsx3
import pandas as pd
import os
from datetime import datetime

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize TTS engine (pyttsx3)
tts = pyttsx3.init()
tts.setProperty("rate", 170)  # speaking speed
tts.setProperty("volume", 1.0)


def _speak(text):
    """Speak text asynchronously (non-blocking typical behavior)."""
    try:
        tts.say(text)
        tts.runAndWait()
    except Exception:
        # fallback: ignore TTS errors so app doesn't crash
        pass


def analyze_expression(frame):
    """Return dominant emotion string using DeepFace (robust to failures)."""
    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        # result may be dict or list depending on version; handle both
        if isinstance(result, list):
            result = result[0]
        emotion = result.get("dominant_emotion") or result.get("dominant_emotion", "Unknown")
        return emotion.capitalize()
    except Exception:
        return "Unknown"


def analyze_posture(frame, pose):
    """Return 'Good' or 'Incorrect' based on simple shoulder-hip alignment."""
    try:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return "Unknown"
        landmarks = results.pose_landmarks.landmark
        # average y positions (normalized coordinates)
        shoulder_y = (
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ) / 2
        hip_y = (
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        ) / 2
        # if shoulders much higher than hips -> good; if too close -> slouch / bad
        if abs(shoulder_y - hip_y) > 0.08:
            return "Good"
        else:
            return "Incorrect"
    except Exception:
        return "Unknown"


def save_session_summary(summary, path="data/session_summaries.csv"):
    """Append session summary (dict) to CSV (create file with header if missing)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([summary])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)


def detect_webcam_feed(alert_thresholds=None, max_duration_seconds=None):
    """
    Start webcam loop; provide voice alerts and return session summary dict after exit.
    - alert_thresholds: dict for when to alert, e.g. {'consecutive_bad_posture': 10, 'emotion_alert': ['tired','sad']}
    - max_duration_seconds: optional auto-stop length
    Returns session_summary dict.
    """
    if alert_thresholds is None:
        alert_thresholds = {"consecutive_bad_posture": 8, "emotion_alert": ["tired", "sad", "angry", "fear"]}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0).")

    # counters & stats
    total_frames = 0
    posture_good_count = 0
    posture_bad_count = 0
    posture_unknown_count = 0
    emotion_counts = {}
    last_posture_alert_time = 0
    consecutive_bad_posture = 0

    start_time = time.time()
    session_start = datetime.now().isoformat(timespec="seconds")

    # Pose object reused for each frame for performance
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            # analyze posture and emotion (emotion is slower - consider running less frequently)
            posture = analyze_posture(frame, pose)
            emotion = analyze_expression(frame) if total_frames % 5 == 0 else "Skip"  # emotion every 5 frames

            # update counts
            if posture == "Good":
                posture_good_count += 1
                consecutive_bad_posture = 0
            elif posture == "Incorrect":
                posture_bad_count += 1
                consecutive_bad_posture += 1
            else:
                posture_unknown_count += 1

            if emotion != "Skip":
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            # voice alerts for posture (if bad for N consecutive frames)
            now = time.time()
            if consecutive_bad_posture >= alert_thresholds.get("consecutive_bad_posture", 8):
                # avoid spamming: only alert if >5s since last alert
                if now - last_posture_alert_time > 5:
                    _speak("Please straighten your back and keep your shoulders aligned.")
                    last_posture_alert_time = now
                    consecutive_bad_posture = 0  # reset to avoid repeated alerts

            # voice alert for emotion if detected and matches list
            if emotion != "Skip":
                dominant_lower = emotion.lower()
                if dominant_lower in alert_thresholds.get("emotion_alert", []):
                    # speak once per 8 seconds for emotions
                    if now - last_posture_alert_time > 8:
                        _speak(f"I notice you seem {emotion}. Consider taking a short break.")
                        last_posture_alert_time = now

            # display overlay text
            display_emotion = emotion if emotion != "Skip" else ""
            text = f"Posture: {posture}  {display_emotion}"
            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 200, 30), 2)

            # draw pose landmarks for feedback
            if pose and hasattr(results := pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), "pose_landmarks") and results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("AI Fitness Coach (press q to end)", frame)

            # quit logic
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if max_duration_seconds and (time.time() - start_time) > max_duration_seconds:
                break

    cap.release()
    cv2.destroyAllWindows()

    # compose session summary
    top_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "Unknown"
    duration = int(time.time() - start_time)
    summary = {
        "session_start": session_start,
        "duration_seconds": duration,
        "frames": total_frames,
        "posture_good": posture_good_count,
        "posture_bad": posture_bad_count,
        "posture_unknown": posture_unknown_count,
        "dominant_emotion": top_emotion,
        "emotion_counts": str(emotion_counts),
    }

    # save to CSV for later viewing/analytics
    try:
        save_session_summary(summary)
    except Exception:
        # ignore saving errors to avoid crashing
        pass

    return summary
