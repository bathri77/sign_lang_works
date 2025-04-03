import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import threading
import time
from collections import deque
import os

# ======= CONFIGURATION =======
# Filtering settings
HISTORY_LENGTH = 15  # Frames to keep in history
STABILITY_NEEDED = 0.7  # 70% agreement needed for stable prediction
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to consider a prediction
SPEECH_INTERVAL = 2.0  # Seconds between spoken outputs
DEBOUNCE_FRAMES = 5  # Frames needed for a new gesture to be confirmed

# Hand detection settings
DETECTION_CONFIDENCE = 0.8  # Hand detector confidence
MIN_HAND_SIZE = 60  # Minimum pixel width/height for valid hand
MAX_HAND_SIZE = 350  # Maximum pixel width/height for valid hand
MARGIN_FACTOR = 0.25  # Extra margin around hand (percentage of hand size)

# Image processing
APPLY_HISTOGRAM_EQ = True  # Apply histogram equalization for better contrast
APPLY_BLUR = True  # Apply slight blur to reduce noise
BLUR_KERNEL_SIZE = 3  # Size of blur kernel
APPLY_THRESHOLD = False  # Apply adaptive thresholding (experimental)
BRIGHTNESS_CORRECTION = 10  # Increase image brightness by this value

# ======= INITIALIZATION =======
# Initialize text-to-speech in a way that minimizes blocking
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Queue for speech commands-
speech_queue = deque()


def speech_worker():
    """Separate thread for text-to-speech to prevent blocking"""
    while True:
        if speech_queue:
            text = speech_queue.popleft()
            engine.say(text)
            engine.runAndWait()
        time.sleep(0.1)


# Start speech thread
threading.Thread(target=speech_worker, daemon=True).start()


def speak_text(text):
    """Add text to speech queue"""
    speech_queue.append(text)


# Kalman filter for prediction smoothing
class KalmanFilter:
    def __init__(self, n_states, n_measurements, dt=1.0):
        self.kalman = cv2.KalmanFilter(n_states, n_measurements)
        self.kalman.measurementMatrix = np.eye(n_measurements, n_states, dtype=np.float32)
        self.kalman.transitionMatrix = np.eye(n_states, dtype=np.float32)
        self.kalman.processNoiseCov = 1e-4 * np.eye(n_states, dtype=np.float32)
        self.kalman.measurementNoiseCov = 1e-3 * np.eye(n_measurements, dtype=np.float32)
        self.kalman.errorCovPost = np.eye(n_states, dtype=np.float32)

    def update(self, measurement):
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return prediction


# Initialize Kalman filter for prediction smoothing
kalman = KalmanFilter(n_states=len(["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]),
                      n_measurements=len(["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]))

# Print configuration banner so user knows enhancements are active
print("\n" + "=" * 50)
print("ADVANCED SIGN LANGUAGE RECOGNITION SYSTEM")
print("=" * 50)
print(f"Stability filter: {HISTORY_LENGTH} frames with {STABILITY_NEEDED * 100}% agreement")
print(f"Enhancement: {'✓' if APPLY_HISTOGRAM_EQ else '✗'} Histogram equalization")
print(f"Enhancement: {'✓' if APPLY_BLUR else '✗'} Noise reduction")
print(f"Enhancement: {'✓' if APPLY_THRESHOLD else '✗'} Adaptive thresholding")
print("=" * 50 + "\n")

# ======= CAMERA SETUP =======
cap = cv2.VideoCapture(0)

# Set increased resolution if camera supports it
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize HandDetector with better parameters
detector = HandDetector(maxHands=1, detectionCon=DETECTION_CONFIDENCE)
classifier = Classifier("converted_keras/keras_model.h5", "converted_keras/labels.txt")

# Image processing parameters
offset = 20
imgSize = 300

# Labels from model
labels = ["Hello", "Thankyou", "Please", "Yes", "okay"]


# State tracking
prediction_history = deque(maxlen=HISTORY_LENGTH)
prediction_confidence = deque(maxlen=HISTORY_LENGTH)
current_stable_sign = None
last_speech_time = time.time()
debounce_counter = 0
previous_prediction = None
gesture_start_time = None

# For FPS calculation
frame_times = deque(maxlen=30)

# ======= MAIN LOOP =======
try:
    while True:
        start_time = time.time()

        # Capture frame
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            continue

        # Make a copy for output display
        imgOutput = img.copy()

        # Find hands
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Validate hand size
            if w < MIN_HAND_SIZE or h < MIN_HAND_SIZE or w > MAX_HAND_SIZE or h > MAX_HAND_SIZE:
                # Hand too small/large - likely a detection error
                cv2.putText(imgOutput, "Invalid hand size", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue

            # Ensure coordinates are within frame boundaries
            if x < 0: x = 0
            if y < 0: y = 0

            try:
                # Calculate margins with extra space
                margin = int(max(w, h) * MARGIN_FACTOR)

                # Create white background image
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                # Apply proper boundary checks for the crop
                y_min = max(0, y - offset - margin)
                y_max = min(img.shape[0], y + h + offset + margin)
                x_min = max(0, x - offset - margin)
                x_max = min(img.shape[1], x + w + offset + margin)

                # Crop the hand region
                imgCrop = img[y_min:y_max, x_min:x_max]

                # Check if cropped image is valid
                if imgCrop.size == 0 or imgCrop.shape[0] <= 0 or imgCrop.shape[1] <= 0:
                    continue

                # ======= IMAGE ENHANCEMENT =======
                # Apply image preprocessing for better recognition
                if APPLY_HISTOGRAM_EQ:
                    # Convert to YUV and equalize the Y channel only
                    imgYUV = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2YUV)
                    imgYUV[:, :, 0] = cv2.equalizeHist(imgYUV[:, :, 0])
                    imgCrop = cv2.cvtColor(imgYUV, cv2.COLOR_YUV2BGR)

                # Increase brightness slightly
                imgCrop = cv2.convertScaleAbs(imgCrop, alpha=1.0, beta=BRIGHTNESS_CORRECTION)

                if APPLY_BLUR:
                    # Apply slight Gaussian blur to reduce noise
                    imgCrop = cv2.GaussianBlur(imgCrop, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

                if APPLY_THRESHOLD:
                    # Convert to grayscale and apply adaptive thresholding
                    imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                    imgThresh = cv2.adaptiveThreshold(imgGray, 255,
                                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY, 11, 2)
                    imgCrop = cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR)

                # Calculate aspect ratio
                aspectRatio = h / w

                # Resize keeping aspect ratio
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

                # ======= PREDICTION =======
                # Get prediction from classifier
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Calculate max confidence
                max_conf = max(prediction)

                # Only consider predictions with good confidence
                if max_conf > CONFIDENCE_THRESHOLD:
                    # Store prediction and its confidence
                    prediction_history.append(index)
                    prediction_confidence.append(max_conf)

                    # Count occurrences of each prediction in history
                    if prediction_history:
                        from collections import Counter

                        counts = Counter(prediction_history)

                        # Find the most common and second most common predictions
                        most_common = counts.most_common(2)

                        if most_common:
                            stable_index = most_common[0][0]
                            stability = counts[stable_index] / len(prediction_history)

                            # Only use the stable prediction if it's stable enough
                            if stability >= STABILITY_NEEDED:
                                # Check if prediction is new
                                new_stable_sign = labels[stable_index]

                                # If we have a new sign, start debounce counter
                                if new_stable_sign != current_stable_sign:
                                    if debounce_counter < DEBOUNCE_FRAMES:
                                        debounce_counter += 1
                                    else:
                                        # New stable sign detected
                                        current_stable_sign = new_stable_sign
                                        debounce_counter = 0

                                        # Record the start time of this gesture
                                        if gesture_start_time is None:
                                            gesture_start_time = time.time()
                                else:
                                    # Reset debounce for consistent prediction
                                    debounce_counter = 0

                                # Speak the sign based on interval
                                current_time = time.time()

                                # Calculate how long this gesture has been held
                                gesture_duration = 0
                                if gesture_start_time:
                                    gesture_duration = current_time - gesture_start_time

                                # Speak the sign if enough time has passed since last speech
                                if (current_time - last_speech_time > SPEECH_INTERVAL and
                                        current_stable_sign is not None):
                                    speak_text(current_stable_sign)
                                    last_speech_time = current_time
                                    print(
                                        f"Speaking: {current_stable_sign} (Conf: {max_conf:.2f}, Stability: {stability:.1f}%, Held for: {gesture_duration:.1f}s)")

                                # Draw UI elements for stable prediction
                                cv2.rectangle(imgOutput, (x - offset, y - offset - 100),
                                              (x - offset + 400, y - offset),
                                              (0, 255, 0), cv2.FILLED)

                                # Show confidence and stability info
                                text_color = (0, 0, 0)  # Black text

                                # Show held duration if significant
                                duration_text = f" Held: {gesture_duration:.1f}s" if gesture_duration > 1.0 else ""

                                confidence_text = f"{current_stable_sign} ({max_conf:.2f} / {stability:.0%}){duration_text}"
                                cv2.putText(imgOutput, confidence_text, (x, y - 60),
                                            cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 2)

                                # Show "STABLE" indicator with green box
                                cv2.putText(imgOutput, "STABLE", (x, y - 30),
                                            cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 2)
                                cv2.rectangle(imgOutput, (x - offset, y - offset),
                                              (x + w + offset, y + h + offset), (0, 255, 0), 4)
                            else:
                                # Reset gesture timing for unstable predictions
                                gesture_start_time = None

                                # Draw UI for unstable prediction
                                cv2.rectangle(imgOutput, (x - offset, y - offset - 100),
                                              (x - offset + 400, y - offset),
                                              (0, 255, 255), cv2.FILLED)

                                # Determine the best guess from current values
                                best_guess = labels[stable_index]
                                confidence_text = f"{best_guess} ({max_conf:.2f} / {stability:.0%})"

                                cv2.putText(imgOutput, confidence_text, (x, y - 60),
                                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
                                cv2.putText(imgOutput, "STABILIZING...", (x, y - 30),
                                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
                                cv2.rectangle(imgOutput, (x - offset, y - offset),
                                              (x + w + offset, y + h + offset), (0, 255, 255), 4)

                    # Show preprocessing debug windows
                    cv2.imshow('Processed Hand', imgCrop)
                    cv2.imshow('Classifier Input', imgWhite)
                else:
                    # Low confidence prediction
                    gesture_start_time = None
                    cv2.rectangle(imgOutput, (x - offset, y - offset),
                                  (x + w + offset, y + h + offset), (0, 0, 255), 4)
                    cv2.putText(imgOutput, f"Low confidence: {max_conf:.2f}", (x, y - 30),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error processing hand: {e}")
        else:
            # Reset when no hand detected
            current_stable_sign = None
            gesture_start_time = None

        # ======= PERFORMANCE MONITORING =======
        # Calculate and show FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        avg_frame_time = sum(frame_times) / len(frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        # Add FPS counter to display
        cv2.putText(imgOutput, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show current stable sign in top-right corner for easy viewing
        if current_stable_sign:
            cv2.putText(imgOutput, f"Sign: {current_stable_sign}",
                        (imgOutput.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display output
        cv2.imshow('Sign Language Recognition', imgOutput)

        # Check for exit key
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")