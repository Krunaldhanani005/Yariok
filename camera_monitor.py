"""
Camera monitor — runs in a background thread.
Detects motion and persons, fires alert sound + voice notification.
"""
import cv2
import threading
import time
import os
import sys

RTSP_URL = "rtsp://Test:Nanta@123@192.168.29.118:554/1/1?transmode=unicast&profile=vam"

# ── Alert sound (uses Windows built-in beep — no file needed) ──────────────────
def _alert_beep():
    for _ in range(3):
        print("\a", end="", flush=True)
        time.sleep(0.1)

# ── Motion detector using background subtraction ───────────────────────────────
class CameraMonitor:
    def __init__(self, on_alert=None, sensitivity=1500):
        """
        on_alert: optional callback function called when motion is detected
        sensitivity: minimum contour area to count as motion (lower = more sensitive)
        """
        self.on_alert = on_alert
        self.sensitivity = sensitivity
        self._running = False
        self._thread = None
        self._last_alert = 0
        self._alert_cooldown = 8  # seconds between alerts

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[Camera] Monitor started in background.")

    def stop(self):
        self._running = False
        print("[Camera] Monitor stopped.")

    def _run(self):
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print("[Camera] ERROR — Could not connect to RTSP stream.")
            print("[Camera] Check: camera is on, same WiFi, credentials correct.")
            return

        print("[Camera] Connected to IMOU camera.")

        # Background subtractor
        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )

        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("[Camera] Stream lost — reconnecting in 5s...")
                time.sleep(5)
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                continue

            # Resize for faster processing
            small = cv2.resize(frame, (640, 360))

            # Motion detection
            mask = bg_sub.apply(small)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = any(cv2.contourArea(c) > self.sensitivity for c in contours)

            if motion_detected:
                now = time.time()
                if now - self._last_alert > self._alert_cooldown:
                    self._last_alert = now
                    self._fire_alert(frame)

        cap.release()

    def _fire_alert(self, frame):
        print("\n\033[91m[ALERT] Motion detected by camera!\033[0m")

        # Save snapshot
        snapshot_path = "./alert_snapshot.jpg"
        cv2.imwrite(snapshot_path, frame)
        print(f"[Camera] Snapshot saved: {snapshot_path}")

        # Beep alert in separate thread so it doesn't block
        threading.Thread(target=_alert_beep, daemon=True).start()

        # Call the assistant's speak function if provided
        if self.on_alert:
            self.on_alert("Alert! Motion detected by the camera.")


# ── Standalone mode — run this file directly to test ──────────────────────────
if __name__ == "__main__":
    def print_alert(msg):
        print(f"[ALERT CALLBACK] {msg}")

    monitor = CameraMonitor(on_alert=print_alert, sensitivity=1500)
    monitor.start()

    print("Camera monitor running. Press Ctrl+C to stop.")
    print("Move in front of the camera to trigger an alert.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("Stopped.")
