import cv2
import mediapipe as mp
import numpy as np

class EyeTracker:
    def __init__(self, width=640, height=480):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, width)
        self.cap.set(4, height)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.width = width
        self.height = height

    def get_gaze_mask(self, image_size=128):
        success, frame = self.cap.read()
        if not success:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        # Blank mask (same as webcam resolution)
        gaze_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * self.width), int(lm.y * self.height)
                    cv2.circle(gaze_mask, (x, y), 3, 255, -1)

        # Smooth mask
        gaze_mask = cv2.GaussianBlur(gaze_mask, (15, 15), 0)

        # Resize mask to match model input
        gaze_resized = cv2.resize(gaze_mask, (image_size, image_size))
        gaze_resized = gaze_resized / 255.0

        return gaze_resized

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
