# 📌 Step 1: Image Preprocessing Module
#
# In this step, we create a reusable preprocessing function.
#
# This module will:
# - Load image
# - Convert to RGB
# - Resize image
# - Remove noise
#
# This makes our system modular and professional.
import cv2
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Error: Image not loaded")
        return None

    img = cv2.resize(img, (800, 800))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb


# ✅ Get correct base path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "raw")

# 🔍 Check files
files = os.listdir(data_path)
print("Files found:", files)

# ✅ Pick first image automatically
image_path = os.path.join(data_path, files[0])

processed_img = preprocess_image(image_path)

if processed_img is not None:
    plt.imshow(processed_img)
    plt.title("Processed Image")
    plt.axis("off")
    plt.show()