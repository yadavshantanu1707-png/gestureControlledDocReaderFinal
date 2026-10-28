"""
Gesture-Controlled Document Reader
===================================
Control a PDF viewer using hand gestures via your webcam.

Gestures:
    1 finger  (index)       -> Next Page
    2 fingers (peace)       -> Previous Page
    Thumb only              -> Zoom In
    Fist (0 fingers)        -> Zoom Out
    Open palm (5 fingers)   -> Scroll Up
    3 fingers               -> Scroll Down
    4 fingers               -> Exit

Keyboard:
    N  -> Next Page
    P  -> Previous Page
    Q  -> Quit

Dependencies:
    pip install opencv-python mediapipe PyMuPDF numpy
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pymupdf as fitz  # PyMuPDF v2
import time
import sys
import os
import tkinter as tk
from tkinter import filedialog


def resource_path(relative_path):
    """Get absolute path to resource - works for dev and PyInstaller exe."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)


# ---------------------------------------------------------------------------
#  Hand Landmark Connections (for drawing skeleton)
# ---------------------------------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle  (fixed: 0->9 via 5->9)
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


# ---------------------------------------------------------------------------
#  Gesture Detector  (MediaPipe Tasks API — v1.0+)
# ---------------------------------------------------------------------------

class GestureDetector:
    """Detects hand gestures using the MediaPipe Tasks HandLandmarker."""

    GESTURE_MAP = {
        "none":       ("No hand detected",  (150, 150, 150)),
        "point":      (">> Next Page",       (0, 255, 0)),
        "peace":      ("<< Prev Page",       (0, 255, 255)),
        "thumbs_up":  ("[+] Zoom In",        (255, 200, 0)),
        "fist":       ("[-] Zoom Out",       (0, 150, 255)),
        "open_palm":  ("^ Scroll Up",        (255, 100, 100)),
        "three":      ("v Scroll Down",      (100, 100, 255)),
        "four":       ("[X] EXIT",           (0, 0, 255)),
        "unknown":    ("...",                (150, 150, 150)),
    }

    # Landmark indices
    THUMB_TIP = 4;  THUMB_IP = 3
    INDEX_TIP = 8;  INDEX_PIP = 6
    MIDDLE_TIP = 12; MIDDLE_PIP = 10
    RING_TIP = 16;  RING_PIP = 14
    PINKY_TIP = 20; PINKY_PIP = 18

    def __init__(self):
        model_path = resource_path("hand_landmarker.task")
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at: {model_path}")
            print("Download it from: https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            sys.exit(1)

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    # ---- public ---------------------------------------------------------

    def detect(self, frame):
        """Return (gesture_name, annotated_frame)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)

        gesture = "none"

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name  # "Left" or "Right"

            # Draw skeleton on frame
            self._draw_landmarks(frame, landmarks)

            # Classify
            fingers = self._finger_states(landmarks, handedness)
            gesture = self._classify(fingers)

        return gesture, frame

    def release(self):
        self.detector.close()

    # ---- private --------------------------------------------------------

    def _draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks and connections on the frame."""
        h, w = frame.shape[:2]
        points = []
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            points.append((px, py))
            cv2.circle(frame, (px, py), 4, (0, 230, 255), -1)
            cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)

        for i, j in HAND_CONNECTIONS:
            if i < len(points) and j < len(points):
                cv2.line(frame, points[i], points[j], (200, 200, 200), 1, cv2.LINE_AA)

    def _finger_states(self, landmarks, handedness):
        """Return [thumb, index, middle, ring, pinky] as 0/1."""
        lm = landmarks
        fingers = []

        # Thumb — horizontal check
        tip_x = lm[self.THUMB_TIP].x
        ip_x = lm[self.THUMB_IP].x
        if handedness == "Right":
            fingers.append(1 if tip_x > ip_x else 0)
        else:
            fingers.append(1 if tip_x < ip_x else 0)

        # Index, Middle, Ring, Pinky — vertical check (tip above PIP = up)
        for tip, pip in [(self.INDEX_TIP, self.INDEX_PIP),
                         (self.MIDDLE_TIP, self.MIDDLE_PIP),
                         (self.RING_TIP, self.RING_PIP),
                         (self.PINKY_TIP, self.PINKY_PIP)]:
            fingers.append(1 if lm[tip].y < lm[pip].y else 0)

        return fingers

    @staticmethod
    def _classify(fingers):
        thumb, index, middle, ring, pinky = fingers
        total = sum(fingers)

        if total == 0:
            return "fist"
        if total == 5:
            return "open_palm"
        if total == 1 and thumb:
            return "thumbs_up"
        if total == 1 and index:
            return "point"
        if total == 2 and index and middle:
            return "peace"
        if total == 3 and index and middle and ring:
            return "three"
        if total == 4 and not thumb:
            return "four"
        return "unknown"


# ---------------------------------------------------------------------------
#  PDF Viewer
# ---------------------------------------------------------------------------

class PDFViewer:
    """Loads a PDF and renders pages as OpenCV images."""

    def __init__(self, path):
        self.doc = fitz.open(path)
        self.page_num = 0
        self.total_pages = len(self.doc)
        self.zoom = 1.5
        self.scroll_y = 0
        self._cache_page = -1
        self._cache_zoom = -1
        self._cache_img = None

    # ---- navigation -----------------------------------------------------

    def next_page(self):
        if self.page_num < self.total_pages - 1:
            self.page_num += 1
            self.scroll_y = 0

    def prev_page(self):
        if self.page_num > 0:
            self.page_num -= 1
            self.scroll_y = 0

    def zoom_in(self):
        self.zoom = min(4.0, round(self.zoom + 0.1, 2))

    def zoom_out(self):
        self.zoom = max(0.5, round(self.zoom - 0.1, 2))

    def scroll_up(self, px=40):
        self.scroll_y = max(0, self.scroll_y - px)

    def scroll_down(self, px=40):
        self.scroll_y += px

    # ---- rendering ------------------------------------------------------

    def render(self, disp_w=960, disp_h=720):
        """Return a BGR numpy image of size (disp_h, disp_w, 3)."""
        if self._cache_page != self.page_num or self._cache_zoom != self.zoom:
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = self.doc[self.page_num].get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4
                               else cv2.COLOR_RGB2BGR)
            self._cache_img = img
            self._cache_page = self.page_num
            self._cache_zoom = self.zoom

        src = self._cache_img
        src_h, src_w = src.shape[:2]

        # Scale to fit display width if needed
        if src_w > disp_w:
            scale = disp_w / src_w
            src = cv2.resize(src, (disp_w, int(src_h * scale)),
                             interpolation=cv2.INTER_AREA)
            src_h, src_w = src.shape[:2]

        # Clamp scroll
        max_scroll = max(0, src_h - disp_h)
        self.scroll_y = min(self.scroll_y, max_scroll)

        # Vertical viewport
        y0 = int(self.scroll_y)
        y1 = min(y0 + disp_h, src_h)
        crop = src[y0:y1]

        # Dark canvas
        canvas = np.full((disp_h, disp_w, 3), 40, dtype=np.uint8)

        # Center crop on canvas
        ch, cw = crop.shape[:2]
        ox = max(0, (disp_w - cw) // 2)
        canvas[0:ch, ox:ox + cw] = crop

        return canvas

    def close(self):
        self.doc.close()


# ---------------------------------------------------------------------------
#  HUD Overlay
# ---------------------------------------------------------------------------

def draw_hud(frame, gesture, page_num, total_pages, zoom):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"Page {page_num + 1} / {total_pages}   Zoom {zoom:.1f}x",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    label, color = GestureDetector.GESTURE_MAP.get(gesture, ("...", (200, 200, 200)))
    cv2.putText(frame, label, (w - 280, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Bottom legend
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 32), (w, h), (25, 25, 25), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    legend = "1F:Next | 2F:Prev | Thumb:Zoom+ | Fist:Zoom- | Palm:Up | 3F:Down | 4F:Quit"
    cv2.putText(frame, legend, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)


# ---------------------------------------------------------------------------
#  File Dialog
# ---------------------------------------------------------------------------

def pick_pdf():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a PDF Document",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
    )
    root.destroy()
    return path if path else None


# ---------------------------------------------------------------------------
#  Main Loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  Gesture-Controlled Document Reader")
    print("=" * 50)
    print()

    # 1. Pick a PDF
    print("Opening file dialog ...")
    pdf_path = pick_pdf()
    if not pdf_path:
        print("No file selected — exiting.")
        return

    print(f"Loading: {pdf_path}")

    # 2. Init components
    viewer = PDFViewer(pdf_path)
    detector = GestureDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    DISP_W, DISP_H = 960, 720
    CAM_W, CAM_H = 240, 180

    prev_gesture = "none"
    last_action = 0.0
    COOLDOWN_NAV = 0.8
    COOLDOWN_CONT = 0.12

    print("Ready! Show hand gestures to your webcam.")
    print("Press Q to quit.\n")

    while True:
        ok, cam = cap.read()
        if not ok:
            break

        cam = cv2.flip(cam, 1)
        gesture, cam_ann = detector.detect(cam)
        now = time.time()

        # --- act on gesture ---
        if gesture not in ("none", "unknown"):
            if gesture in ("open_palm", "three", "fist", "thumbs_up"):
                if now - last_action > COOLDOWN_CONT:
                    {"open_palm": viewer.scroll_up,
                     "three":     viewer.scroll_down,
                     "thumbs_up": viewer.zoom_in,
                     "fist":      viewer.zoom_out}[gesture]()
                    last_action = now
            else:
                if gesture != prev_gesture and now - last_action > COOLDOWN_NAV:
                    if gesture == "point":
                        viewer.next_page()
                    elif gesture == "peace":
                        viewer.prev_page()
                    elif gesture == "four":
                        print("Exit gesture detected.")
                        break
                    last_action = now

        prev_gesture = gesture

        # --- compose display ---
        pdf_frame = viewer.render(DISP_W, DISP_H)

        # PiP webcam overlay (bottom-right)
        pip = cv2.resize(cam_ann, (CAM_W, CAM_H))
        cv2.rectangle(pip, (0, 0), (CAM_W - 1, CAM_H - 1), (0, 200, 255), 2)
        y1 = DISP_H - CAM_H - 10
        x1 = DISP_W - CAM_W - 10
        pdf_frame[y1:y1 + CAM_H, x1:x1 + CAM_W] = pip

        draw_hud(pdf_frame, gesture, viewer.page_num,
                 viewer.total_pages, viewer.zoom)

        cv2.imshow("Gesture Document Reader", pdf_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key == ord("n"):
            viewer.next_page()
        elif key == ord("p"):
            viewer.prev_page()

    # Cleanup
    detector.release()
    cap.release()
    viewer.close()
    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()
