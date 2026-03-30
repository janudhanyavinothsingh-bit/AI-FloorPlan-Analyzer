# 📌 Step 4: Generating Structured Output

# We convert detected rooms into structured JSON format.

# This makes the system:
# Industry-ready
# API-friendly
import json

# ------------------------------------------
# Room Classification Function
# ------------------------------------------
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


# ------------------------------------------
# Generate Output Function
# ------------------------------------------
def generate_output(rooms):
    result = []

    for i, room in enumerate(rooms):
        room_data = {
            "room_id": i,
            "area": room["area"],
            "type": classify_room(room["area"])
        }
        result.append(room_data)

    return result


# ------------------------------------------
# TEST (Important 🔥)
# ------------------------------------------
if __name__ == "__main__":

    # Sample input (like your detection output)
    rooms = [
        {"area": 60000},
        {"area": 35000},
        {"area": 20000},
        {"area": 8000}
    ]

    output = generate_output(rooms)

    # Print nicely
    print(json.dumps(output, indent=4))