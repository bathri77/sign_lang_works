import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

# Create directory if it doesn't exist
folder = "Data/okay"
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Make sure the crop region is within the image boundaries
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        # Crop the hand region
        imgCrop = img[y1:y2, x1:x2]

        # Check if the cropped image is valid
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            # Calculate aspect ratio
            aspectRatio = h / w

            if aspectRatio > 1:
                # Hand is taller than wide
                k = imgSize / h
                wCal = math.ceil(k * w)
                if wCal > 0:  # Ensure width is positive
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    # Place the resized image on the white background
                    imgWhite[:, wGap:wCal + wGap] = imgResize

                    # Display the images
                    cv2.imshow('ImageCrop', imgCrop)
                    cv2.imshow('ImageWhite', imgWhite)
            else:
                # Hand is wider than tall
                k = imgSize / w
                hCal = math.ceil(k * h)
                if hCal > 0:  # Ensure height is positive
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    # Place the resized image on the white background
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                    # Display the images
                    cv2.imshow('ImageCrop', imgCrop)
                    cv2.imshow('ImageWhite', imgWhite)

    # Display the main camera feed
    cv2.imshow('Image', img)

    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord("s"):
        # Save the processed hand image
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")
    elif key == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()