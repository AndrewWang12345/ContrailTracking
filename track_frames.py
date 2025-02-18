import os
import cv2
import json

# Paths
VIDEO_FOLDER = "video/"
ANNOTATION_FOLDER = "coco_output/"
OUTPUT_FOLDER = "datasets/"  # Folder to store extracted datasets

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get list of videos
video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]

for video_file in video_files:
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    annotation_path = os.path.join(ANNOTATION_FOLDER, f"{video_file}_annotations.json")

    if not os.path.exists(annotation_path):
        print(f"⚠️ No annotation file found for {video_file}, skipping...")
        continue

    # Load COCO JSON annotation
    with open(annotation_path, "r") as f:
        coco_data = json.load(f)

    # Collect frame indices that have annotations
    annotated_frame_indices = set()
    for image_info in coco_data["images"]:
        frame_index = image_info.get("frame_index")
        if frame_index is not None:
            annotated_frame_indices.add(frame_index)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video {video_file}")
        continue

    # Create output folder for datasets from this video
    video_name = os.path.splitext(video_file)[0]
    video_output_folder = os.path.join(OUTPUT_FOLDER, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # ✅ Save only annotated frames
        if frame_idx in annotated_frame_indices:
            frame_filename = f"{video_name}_frame_{frame_idx}.jpg"
            frame_path = os.path.join(video_output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"✅ Saved annotated frame {frame_idx} → {frame_filename}")

        frame_idx += 1

    cap.release()

print("🎉 Annotated frame extraction complete!")
