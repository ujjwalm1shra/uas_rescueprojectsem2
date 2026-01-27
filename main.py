import cv2
import numpy as np
import os

# ================= CONFIG =================
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# ================= PRIORITY TABLES =================
AGE_PRIORITY = {
    "Star": 3,
    "Triangle": 2,
    "Square": 1
}

EMERGENCY_PRIORITY = {
    "Red": 3,
    "Yellow": 2,
    "Green": 1
}

CAMP_CAPACITY = {
    "Blue": 4,
    "Pink": 3,
    "Grey": 2
}

# ================= HELPERS =================
def detect_shape(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    v = len(approx)

    if v == 3:
        return "Triangle"
    elif v == 4:
        return "Square"
    elif v >= 10:
        return "Star"
    elif v >= 6:
        return "Circle"
    return "Unknown"


def detect_color(cnt, hsv):
    mask = np.zeros(hsv.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mask = cv2.erode(mask, np.ones((5,5), np.uint8), 1)

    h, s, v, _ = cv2.mean(hsv, mask=mask)

    if s < 40 and v > 180:
        return "Grey"

    if h < 10 or h > 160:
        return "Red"
    elif 18 < h < 38:
        return "Yellow"
    elif 38 < h < 85:
        return "Green"
    elif 90 < h < 125:
        return "Blue"
    elif 125 < h < 155:
        return "Pink"

    return "Unknown"


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# ================= USER INPUT =================
file_name = input("Enter image name (e.g. 2.png): ").strip()
input_path = os.path.join("images", file_name)


image = cv2.imread(input_path)
if image is None:
    print("Image not found")
    exit()

file_name = os.path.basename(input_path)
output_path = os.path.join(output_folder, file_name)

# ================= IMAGE PROCESSING =================
image = cv2.resize(image, (800, 500))
display = image.copy()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# ---------- LAND MASK ----------
land_lower = np.array([43,110,91])
land_upper = np.array([77,238,161])
land_mask = cv2.inRange(hsv, land_lower, land_upper)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 2)
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

thresh[land_mask == 255] = 0

contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)

display[land_mask == 255] = (0,200,255)

# ---------- RESET ----------
people = []
camps = []

# ---------- DETECT OBJECTS ----------
for cnt in contours:
    if cv2.contourArea(cnt) < 200:
        continue

    shape = detect_shape(cnt)
    color = detect_color(cnt, hsv)

    x,y,w,h = cv2.boundingRect(cnt)
    cx, cy = x + w//2, y + h//2

    cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,255), 1)
    cv2.putText(display, f"{color} {shape}", (x,y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

    if shape in AGE_PRIORITY and color in EMERGENCY_PRIORITY:
        people.append({
            "center": (cx,cy),
            "age_p": AGE_PRIORITY[shape],
            "emg_p": EMERGENCY_PRIORITY[color],
            "base_priority": AGE_PRIORITY[shape]*EMERGENCY_PRIORITY[color]
        })

    if shape == "Circle" and color in CAMP_CAPACITY:
        camps.append({
            "center": (cx,cy),
            "color": color,
            "capacity": CAMP_CAPACITY[color],
            "assigned": []
        })

# ---------- ASSIGNMENT ----------
assignments = []
for p in people:
    for c in camps:
        d = distance(p["center"], c["center"])
        score = p["base_priority"] / (d + 1)
        assignments.append({"p":p, "c":c, "score":score})

assignments.sort(key=lambda x: x["score"], reverse=True)

used = set()
for a in assignments:
    if id(a["p"]) in used:
        continue
    if len(a["c"]["assigned"]) >= a["c"]["capacity"]:
        continue
    a["c"]["assigned"].append(a["p"])
    used.add(id(a["p"]))

# ---------- OUTPUT ----------
print("\n==============================")
print(f"IMAGE: {file_name}")
print("==============================")

camp_scores = []
for c in camps:
    print(f"{c['color']} camp:")
    for p in c["assigned"]:
        print([p["age_p"], p["emg_p"]])
    camp_scores.append(sum(p["base_priority"] for p in c["assigned"]))

ratio = sum(camp_scores)/len(people) if people else 0
print("Camp priority scores:", camp_scores)
print("Image Priority Ratio:", round(ratio,3))

cv2.imwrite(output_path, display)
print(f"\n Output saved to: {output_path}")
