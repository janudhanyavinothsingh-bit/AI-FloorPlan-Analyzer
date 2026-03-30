# 📌 Step 3: Room Classification
# We classify rooms based on area.
# This is a rule-based approach (beginner-friendly),
# which can later be replaced by ML models.
# ==========================================
# 📌 Room Classification Function + Test
# ==========================================

def classify_room(area):
    # Safety check
    if area is None or area <= 0:
        return "Unknown"

    if area > 50000:
        return "Hall"
    elif area > 30000:
        return "Bedroom"
    elif area > 15000:
        return "Kitchen"
    else:
        return "Bathroom"


# ==========================================
# 📌 Test the Function
# ==========================================

if __name__ == "__main__":
    test_areas = [60000, 40000, 20000, 8000, 0]

    for area in test_areas:
        room_type = classify_room(area)
        print(f"Area: {area} → Room Type: {room_type}")