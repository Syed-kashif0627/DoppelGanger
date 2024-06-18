import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
font = cv2.FONT_HERSHEY_DUPLEX

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
aruco_actual_width = 10  # Change this value to the actual width of your ArUco marker
aruco_actual_height = 10  # Change this value to the actual height of your ArUco marker

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_marker_dimensions(corners):
    if len(corners) != 4:
        raise ValueError("Four corners are required to calculate dimensions")

    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]

    width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 100
    height = math.sqrt((x3 - x1)**2 + (y3 - y1)**2) / 100

    vertical_ref = 147 / height
    horizontal_ref = 147 / width

    return vertical_ref, horizontal_ref

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.vertical_ref = 1
        self.horizontal_ref = 1
        self.size_label = "None"
        self.size_ranges = {
            'S': (0, 30),
            'M': (31, 49),
            'L': (50, 80),
            'XL': (81, 100),
            'XXL': (101, float('inf')),
        }
        self.shoulder_size_ranges = {
            '2XS': (38.1, 38.1),
            'XS': (39.37, 39.37),
            'S': (43.18, 43.18),
            'M': (45.72, 45.72),
            'L': (48.26, 48.26),
            'XL': (50.8, 50.8),
            '2XL': (53.34, 53.34),
            '3XL': (55.88, 55.88),
        }
        self.chest_size_ranges = {
            '2XS': (86.36, 86.36),
            'XS': (91.44, 91.44),
            'S': (96.52, 96.52),
            'M': (101.6, 101.6),
            'L': (106.68, 106.68),
            'XL': (111.76, 111.76),
            '2XL': (116.84, 116.84),
            '3XL': (121.92, 121.92),
        }
        self.body_size_ranges = {
            '2XS': (63.5, 63.5),
            'XS': (66.04, 66.04),
            'S': (68.58, 68.58),
            'M': (71.12, 71.12),
            'L': (73.66, 73.66),
            'XL': (76.2, 76.2),
            '2XL': (78.74, 78.74),
            '3XL': (81.28, 81.28),
        }
        self.sleeve_length_ranges = {
            '2XS': (50.8, 50.8),
            'XS': (50.8, 50.8),
            'S': (53.34, 53.34),
            'M': (53.34, 53.34),
            'L': (55.88, 55.88),
            'XL': (59.69, 59.69),
            '2XL': (61, 61),
            '3XL': (61, 61),
        }


    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24",)
        #img = cv2.flip(img, 1)  # Flip the frame horizontally

        corners, marker_ids, _ = cv2.aruco.detectMarkers(img, dictionary)
        if corners:
            for corner, marker_id in zip(corners, marker_ids):
                cv2.polylines(img, [corner.astype(np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)
                corner = corner.reshape(4, 2).astype(int)
                top_right, top_left, bottom_right, bottom_left = corner

                cv2.putText(img, f"id: {marker_id[0]}", top_right, font, 1, (100, 0, 0), 2)
                distance_width, distance_height = calculate_marker_dimensions(corner)
                cv2.putText(img, f"Width: {distance_width:.0f} , Height: {distance_height:.0f}",
                            (top_left[0], top_left[1] - 20), font, 1, (0, 0, 0), 2)

                self.vertical_ref, self.horizontal_ref = calculate_marker_dimensions(corner)

        results = self.pose.process(img)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            head = (landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y)
            shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            waist = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            heel = (landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y)
            left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)
            right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)

            head_to_heel = calculate_distance(head, heel) * self.vertical_ref
            shoulder_to_waist = calculate_distance(shoulder, waist) * self.vertical_ref
            waist_to_heel = calculate_distance(waist, heel) * self.vertical_ref
            left_shoulder_to_right_shoulder = calculate_distance(left_shoulder, right_shoulder) * self.horizontal_ref
            arm_length = calculate_distance(left_wrist, right_wrist) * self.horizontal_ref

            size_label = None
            for size, (lower, upper) in self.size_ranges.items():
                if lower <= shoulder_to_waist <= upper:
                    size_label = size
        
            cv2.putText(img, f'Height: {head_to_heel:.2f} CMs', (10, 30), font, 0.7, (0, 0, 0), 3)
            cv2.putText(img, f'Shirt: {shoulder_to_waist:.2f} CMs -->Your Size: {size_label}', (10, 60), font, 0.7, (0, 0, 0), 3)
            cv2.putText(img, f'Waist to Heel: {waist_to_heel:.2f} CMs', (10, 90), font, 0.7, (0, 0, 0), 3)
            cv2.putText(img, f'Shoulder: {left_shoulder_to_right_shoulder:.2f} CMs', (10, 120), font, 0.7, (0, 0, 0), 3)
            cv2.putText(img, f'Arm Length: {arm_length:.2f} CMs', (10, 150), font, 0.7, (0, 0, 0), 3)

            cv2.putText(img, f"Height: {head_to_heel:.2f} CMs", (10, 30), font, 0.7, (200,200,200), 2)
            cv2.putText(img, f'Shirt: {shoulder_to_waist:.2f} CMs -->Your Size: {size_label}', (10, 60), font, 0.7, (200,200,200), 2)
            cv2.putText(img, f'Waist to Heel: {waist_to_heel:.2f} CMs', (10, 90), font, 0.7, (200,200,200), 2)
            cv2.putText(img, f'Shoulder: {left_shoulder_to_right_shoulder:.2f} CMs', (10, 120), font, 0.7, (200,200,200), 2)
            cv2.putText(img, f'Arm Length: {arm_length:.2f} CMs', (10, 150), font, 0.7, (200,200,200), 2)

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return img

    def get_size_label(self):
        return self.size_label

def main():
    st.title("Doppelganger")
    st.write("Body Measurements Finder")
    st.write("Give access to camera and Start")
    ctx = webrtc_streamer(
        key="body_measurements",
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},  # Disable audio
    )
    
if __name__ == "__main__":
    main()
