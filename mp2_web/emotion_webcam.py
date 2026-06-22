import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# -------------------
# Load trained model
# -------------------
model_path = os.path.abspath(r"D:\PROJECTS\emotion_detector\emotion_model.h5")
model = load_model(model_path)

# FER2013 labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -------------------
# Load Haarcascade for face detection
# -------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------
# Function to capture ONE frame and return emotion
# -------------------
def detect_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    emotion = "No Face"

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]

            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float32") / 255.0

            # Add batch and channel dimensions
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            preds = model.predict(roi_gray, verbose=0)[0]
            emotion = emotion_labels[np.argmax(preds)]

    cap.release()
    return emotion


# -------------------
# Run the program
# -------------------
if __name__ == "__main__":
    emotion = detect_emotion()
    print("Detected Emotion:", emotion)
    input("Press Enter to exit...")