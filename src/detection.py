# 📌 Step 2: Room Detection Module
#This module detects rooms using:
# Color segmentation
# Contour detection
#Each room is identified based on unique colors.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_rooms(img_rgb):
    pixels = img_rgb.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    rooms = []

    for color in unique_colors:
        mask = cv2.inRange(img_rgb, color, color)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 1500:
                x, y, w, h = cv2.boundingRect(cnt)

                rooms.append((x, y, w, h))

    return rooms


# ✅ LOAD IMAGE (same safe method)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "raw")

files = os.listdir(data_path)
image_path = os.path.join(data_path, files[0])

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ✅ CALL FUNCTION
rooms = detect_rooms(img_rgb)

# ✅ PRINT OUTPUT
print("Detected rooms:", len(rooms))

# ✅ DRAW BOUNDING BOXES
for (x, y, w, h) in rooms:
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)

# ✅ SHOW IMAGE
plt.imshow(img_rgb)
plt.title("Detected Rooms")
plt.axis("off")
plt.show()
# 📌 Step 7: Visualizing Detected Rooms

# In this step, we:
# Draw bounding boxes around each detected room
# Add labels (room type + area)

# This helps in:
# Visual validation
# Making the project more professional
import cv2
from classification import classify_room

def draw_rooms(img_rgb, rooms):
    img_copy = img_rgb.copy()

    for room in rooms:
        x, y, w, h = room["x"], room["y"], room["w"], room["h"]
        area = room["area"]
        label = classify_room(area)

        # Draw rectangle
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Put label
        cv2.putText(
            img_copy,
            f"{label}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

    return img_copy