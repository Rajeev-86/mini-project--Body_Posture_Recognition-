import streamlit as st
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import io
import math

# This part is for frontend designing


# This part is backend working 
# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)

# Angle calculation
def calculate_angle(a, b, c):
    # a, b, c are 3 points: [x, y]
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)
    return angle if angle <= 180.0 else 360 - angle

# Posture classification
def classify_pose(landmarks):
    # Get required landmark coordinates
    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

    # Average some key points
    shoulder_y = (shoulder_left[1] + shoulder_right[1]) / 2
    hip_y = (hip_left[1] + hip_right[1]) / 2
    knee_y = (knee_left[1] + knee_right[1]) / 2
    ankle_y = (ankle_left[1] + ankle_right[1]) / 2
    wrist_y = (wrist_left[1] + wrist_right[1]) / 2
    # Average X positions
    shoulder_x = (shoulder_left[0] + shoulder_right[0]) / 2
    hip_x = (hip_left[0] + hip_right[0]) / 2
    ankle_x = (ankle_left[0] + ankle_right[0]) / 2
    wrist_diff_y = abs(wrist_left[1] - shoulder_left[1]) + abs(wrist_right[1] - shoulder_right[1])
    wrist_diff_x = abs(wrist_left[0] - wrist_right[0])
    left_arm_y_diff = abs(wrist_left[1] - shoulder_left[1])
    right_arm_y_diff = abs(wrist_right[1] - shoulder_right[1])

    # Calculate key angles
    knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)
    knee_angle_right = calculate_angle(hip_right, knee_right, ankle_right)
    shoulder_line_angle = calculate_angle(wrist_left, shoulder_left, wrist_right)
    angle_left_arm = calculate_angle(elbow_left, shoulder_left, wrist_left)
    angle_right_arm = calculate_angle(elbow_right, shoulder_right, wrist_right)

    if abs(shoulder_y - hip_y) > 0.05 and 70 <= knee_angle_left <= 110 and 70 <= knee_angle_right <= 110:
        return "Sitting"
    # Heuristic rules
    elif hip_y > knee_y and 40 <= knee_angle_left <= 100 and 40 <= knee_angle_right <= 100:
        return "Squatting"

    elif abs(shoulder_y - hip_y) < 0.05 and abs(hip_y - knee_y) < 0.05:
        return "Lying Down"

    elif wrist_y < shoulder_y:
        return "Arms Raised"

    elif abs(wrist_left[1] - hip_left[1]) < 0.05 and abs(wrist_right[1] - hip_right[1]) < 0.05:
        return "Hands on Hips"
    
    elif (
        abs(shoulder_y - hip_y) > 0.15 and
        knee_angle_left > 165 and knee_angle_right > 165 and
        abs(hip_x - ankle_x) < 0.1 and  # hips aligned over ankles
        wrist_y > shoulder_y and  # arms not raised
        abs(shoulder_x - hip_x) < 0.1  # torso mostly vertical
    ):
        return "Standing"

    else:
        return "Unknown"

# Streamlit UI
st.title("🧍 Posture Recognition System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect pose
    result = pose.process(image_rgb)

    # Draw landmarks
    if result.pose_landmarks:
        annotated_image = image_rgb.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # Classify posture
        posture = classify_pose(result.pose_landmarks.landmark)

        st.subheader("Posture Classification:")
        st.success(posture)

        # Display annotated image
        st.image(annotated_image, caption="Pose Landmarks", use_container_width=True)
    else:
        st.error("No person detected in the image. Try a clearer image.")
