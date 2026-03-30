
import os
import json

from src.preprocessing import preprocess_image
from src.detection import detect_rooms
from src.utils import generate_output
from src.generative_ai import generate_description

# ==========================================
#  PATH SETUP
# ==========================================

data_path = "data/raw/"

files = os.listdir(data_path)

print(" Total Files Found:", len(files))

# ==========================================
# PROCESS IMAGES
# ==========================================

for file in files:
    if file.endswith(".png") or file.endswith(".jpg"):

        print(f"\n Processing: {file}")

        image_path = os.path.join(data_path, file)

        # Step 1: Preprocess
        img = preprocess_image(image_path)

        if img is None:
            print(" Failed to load image")
            continue

        # Step 2: Detect rooms
        rooms = detect_rooms(img)
        print(f" Rooms Detected: {len(rooms)}")

        # Step 3: Generate structured output
        output = generate_output(rooms)

        # Pretty print JSON in console
        print(" Room Details:")
        print(json.dumps(output, indent=4))

        # Step 4: Generate description
        description = generate_description(output)

        print(" Description:")
        print(description)

print("\n PROCESS COMPLETED SUCCESSFULLY!")
