import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def calculate_scale_factor(mask, landmarks, image_height, user_height_cm):
    # Calculate scale factor (cm per pixel) using user's real height.
    ys = np.where(mask > 0.5)

    if len(ys) == 0:
        print("ERROR: No person detected in the image.")
        exit()

    top = ys.min()
    bottom = (
        max(
            landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
        )
        * image_height
    )

    person_height_px = abs(bottom - top)
    cm_per_pixel = user_height_cm / person_height_px

    # print("Top of head (px) :", round(top, 2))
    # print("Bottom / heel (px):", round(bottom, 2))
    # print("Person height (px):", round(person_height_px, 2))
    # print("Scale (cm/px)     :", cm_per_pixel)
    return cm_per_pixel

def calculate_circumference(width_cm, depth_ratio=0.7):
    # Elliptical approximation: C ≈ 2π * sqrt((a² + b²)/2)
    half_width = width_cm / 2
    half_depth = (width_cm * depth_ratio) / 2
    return round(2 * np.pi * np.sqrt((half_width**2 + half_depth**2) / 2), 2)

def calculate_measurements(
    results, mask, image, image_width, image_height, user_height_cm
):
    landmarks = results.pose_landmarks.landmark
    cm_per_pixel = calculate_scale_factor(mask, landmarks, image_height, user_height_cm)

    def px_to_cm(px):
        return round(px * cm_per_pixel, 2)

    measurements = {}

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Shoulder width
    shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * image_width
    shoulder_width_cm = px_to_cm(shoulder_width_px)
    measurements["shoulder_width"] = shoulder_width_cm

    # Hip width
    hip_width_px = abs(left_hip.x - right_hip.x) * image_width
    hip_width_cm = px_to_cm(hip_width_px)
    measurements["hip_width"] = hip_width_cm

    # Chest width - interpolate between shoulder and hip width
    # Chest sits closer to shoulder level, so weight more heavily toward shoulder_width
    chest_width_cm = (
        shoulder_width_cm * 0.85
    )  # chest is typically slightly narrower than shoulder-to-shoulder
    measurements["chest_width"] = round(chest_width_cm, 2)
    measurements["chest_circumference"] = calculate_circumference(chest_width_cm)

    # Waist width - interpolate between shoulder and hip, weighted toward hip since
    # waist sits closer to hip level than chest does
    waist_width_cm = (shoulder_width_cm * 0.4) + (hip_width_cm * 0.6)
    measurements["waist_width"] = round(waist_width_cm, 2)
    measurements["waist_circumference"] = calculate_circumference(waist_width_cm)

    return measurements

if __name__ == "__main__":
    image_path = "sample_photo.jpeg"
    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Could not read image.")
        exit()

    image_height, image_width = image.shape[:2]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)
    segmentation_results = segmentor.process(rgb_image)

    mask = segmentation_results.segmentation_mask

    if not results.pose_landmarks:
        print("ERROR: No pose detected.")
        exit()

    user_height_cm = 152

    measurements = calculate_measurements(
        results, mask, image, image_width, image_height, user_height_cm
    )

    for name, value in measurements.items():
        print(f"{name:<25}: {value:.2f} cm")

    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )
    cv2.imwrite("output.jpg", annotated_image)
    print("\nOutput saved as output.jpg")