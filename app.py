# 📌 Step 10: Building Web Application using Streamlit

# We create a user interface where users can:
# Upload floor plan
# Analyze automatically
# View results visually
import streamlit as st
import cv2
import numpy as np
from collections import Counter

# ==========================================
# Preprocessing
# ==========================================
def preprocess_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Resize while keeping aspect ratio
    max_dim = 800
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb

# ==========================================
# Classification
# ==========================================
def classify_room(area):
    if area is None or area <= 0:
        return "Unknown"
    elif area > 50000:
        return "Hall"
    elif area > 30000:
        return "Bedroom"
    elif area > 15000:
        return "Kitchen"
    else:
        return "Bathroom"

# ==========================================
# Detection
# ==========================================
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
                rooms.append({
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": area,
                    "type": classify_room(area)
                })

    return rooms

# ==========================================
# Description Generator
# ==========================================
def generate_description(rooms):
    types = [room["type"] for room in rooms]
    count = Counter(types)

    parts = []
    for k, v in count.items():
        parts.append(f"{v} {k}" + ("s" if v > 1 else ""))

    return "This floor plan contains " + ", ".join(parts)

# ==========================================
# Streamlit UI
# ==========================================
st.title("AI Floor Plan Analyzer")

uploaded_file = st.file_uploader("Upload Floor Plan Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess
    img_rgb = preprocess_image(uploaded_file)

    # Detect rooms
    rooms = detect_rooms(img_rgb)

    # Draw bounding boxes
    for room in rooms:
        x, y, w, h = room["x"], room["y"], room["w"], room["h"]
        label = room["type"]

        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        y_text = max(y - 10, 10)
        cv2.putText(img_rgb, label, (x, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show image
    st.image(img_rgb, caption="Processed Floor Plan", use_column_width=True)

    # Show results
    st.subheader(f"Total Rooms Detected: {len(rooms)}")

    # Description
    description = generate_description(rooms)
    st.write(description)