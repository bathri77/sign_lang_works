import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import threading
import time
from collections import deque

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech


# Function to speak text in a separate thread
def speak_text(text):
    def speak():
        engine.say(text)
        engine.runAndWait()

    # Run speech in a separate thread to avoid blocking the main loop
    threading.Thread(target=speak).start()


# Improved hand detection with better parameters
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8, minTrackCon=0.5)
classifier = Classifier("converted_keras/keras_model.h5", "converted_keras/labels.txt")
offset = 20
imgSize = 300

labels = ["Hello", "Thankyou", "Please", "Yes", "okay"]

# Timer for speech
last_speech_time = time.time()
speech_interval = 2.0  # seconds between utterances (increased for stability)

# Prediction stabilization
prediction_history = deque(maxlen=10)  # Store last 10 predictions
confidence_threshold = 0.7  # Only consider predictions above this confidence

# For FPS calculation
prev_time = 0
current_time = 0

while True:
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        continue

    imgOutput = img.copy()

    # Find hands with improved detection confidence
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure coordinates are within frame boundaries and hand is large enough
        if x < 0: x = 0
        if y < 0: y = 0

        # Only process if hand is of reasonable size
        if w > 50 and h > 50 and w < 300 and h < 300:  # Filter out too small or too large detections
            try:
                # Add some margin to the bounding box for better results
                margin = int(max(w, h) * 0.2)  # 20% margin
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                # Apply proper boundary checks for the crop
                y_min = max(0, y - offset - margin)
                y_max = min(img.shape[0], y + h + offset + margin)
                x_min = max(0, x - offset - margin)
                x_max = min(img.shape[1], x + w + offset + margin)

                imgCrop = img[y_min:y_max, x_min:x_max]

                # Check if cropped image is valid
                if imgCrop.size == 0 or imgCrop.shape[0] <= 0 or imgCrop.shape[1] <= 0:
                    continue

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                # Get prediction
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Only consider high-confidence predictions
                max_conf = max(prediction)
                if max_conf > confidence_threshold:
                    # Add the prediction to history
                    prediction_history.append(index)

                    # Get the most common prediction from history
                    if prediction_history:
                        # Count occurrences of each prediction
                        from collections import Counter

                        counts = Counter(prediction_history)
                        stable_index = counts.most_common(1)[0][0]

                        # Calculate stability percentage
                        stability = counts[stable_index] / len(prediction_history) * 100

                        # Only use the stable prediction if it's stable enough
                        if stability > 60:  # At least 60% of recent predictions agree
                            current_sign = labels[stable_index]

                            # Speak with interval
                            if current_time - last_speech_time > speech_interval:
                                speak_text(current_sign)
                                last_speech_time = current_time
                                print(f"Speaking: {current_sign} (Confidence: {max_conf:.2f}, Stability: {stability:.1f}%)")

                            # Draw UI elements
                            cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                                          (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                                          cv2.FILLED)

                            # Show confidence and stability info
                            confidence_text = f"{current_sign} - Conf: {max_conf:.2f}, Stbl: {stability:.0f}%"
                            cv2.putText(imgOutput, confidence_text, (x, y - 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                            # Green rectangle for stable prediction
                            cv2.rectangle(imgOutput, (x - offset, y - offset),
                                          (x + w + offset, y + h + offset), (0, 255, 0), 4)
                        else:
                            # Yellow rectangle for unstable prediction
                            cv2.rectangle(imgOutput, (x - offset, y - offset),
                                          (x + w + offset, y + h + offset), (0, 255, 255), 4)

                    cv2.imshow('ImageCrop', imgCrop)
                    cv2.imshow('ImageWhite', imgWhite)
                else:
                    # Red rectangle for low confidence
                    cv2.rectangle(imgOutput, (x - offset, y - offset),
                                  (x + w + offset, y + h + offset), (0, 0, 255), 4)

            except Exception as e:
                print(f"Error processing hand: {e}")

    # Display FPS
    cv2.putText(imgOutput, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Image', imgOutput)
    key = cv2.waitKey(1)

    # Press 'q' to quit
    if key == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()