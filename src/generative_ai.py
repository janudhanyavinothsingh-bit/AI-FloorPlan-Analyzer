# 📌 Step 9: Generative AI Explanation

# We generate human-readable insights from structured data.

# This adds intelligence to the system.
from collections import Counter

# ------------------------------------------
# Generate Description Function
# ------------------------------------------
def generate_description(rooms_output):
    room_types = [room["type"] for room in rooms_output]

    count = Counter(room_types)

    description = "This floor plan contains "

    parts = []
    for k, v in count.items():
        parts.append(f"{v} {k}(s)")

    description += ", ".join(parts)

    return description


# ------------------------------------------
# TEST (Important 🔥)
# ------------------------------------------
if __name__ == "__main__":

    # Sample input (like your previous output)
    rooms_output = [
        {"type": "Hall"},
        {"type": "Bedroom"},
        {"type": "Bedroom"},
        {"type": "Kitchen"},
        {"type": "Bathroom"}
    ]

    result = generate_description(rooms_output)

    print(result)