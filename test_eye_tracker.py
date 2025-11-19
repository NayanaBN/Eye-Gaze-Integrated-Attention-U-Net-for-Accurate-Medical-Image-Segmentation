import cv2
import mediapipe as mp
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # blank heatmap
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                for lm in landmarks.landmark:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(heatmap, (x, y), 2, 255, -1)

        # Convert to BGR heatmap
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Ensure size match
        heatmap_color = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))

        # Overlay
        overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

        cv2.imshow("Eye Tracker", overlay)

        if cv2.waitKey(5) & 0xFF == 27:  # press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
