import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as mask_utils

# replace these paths with the paths to your annotation, video, and output directory
coco_file = "C:\\Users\\joe\\PycharmProjects\\ContrailTracking\\coco_output\\US000F_20241114-319_frames_timelapse.mp4_annotations.json"
video_file = "C:\\Users\\joe\\PycharmProjects\\ContrailTracking\\video\\US000F_20241114-319_frames_timelapse.mp4"
output_video_path = "C:\\Users\\joe\\PycharmProjects\\ContrailTracking\\annotated_video.mp4"

with open(coco_file) as f:
    coco_data = json.load(f)

# Open video
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_file}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# ✅ 1️⃣ Create a mapping of image_id → frame_index
image_id_to_frame_index = {}
for image_info in coco_data["images"]:
    image_id_to_frame_index[image_info["id"]] = image_info["frame_index"]

# ✅ 2️⃣ Create a mapping of frame_index → annotations
frame_annotations = {}
for annotation in coco_data['annotations']:
    image_id = annotation["image_id"]
    frame_index = image_id_to_frame_index.get(image_id, -1)  # Get frame_index from image_id

    if frame_index == -1:
        continue  # Skip if frame index is missing

    if frame_index not in frame_annotations:
        frame_annotations[frame_index] = []
    frame_annotations[frame_index].append(annotation)

# ✅ 3️⃣ Process each frame
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Convert frame to RGB (for overlaying colored masks)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ 4️⃣ Check if this frame has annotations
    if frame_idx in frame_annotations:
        for annotation in frame_annotations[frame_idx]:
            segmentation = annotation['segmentation']

            # ✅ 5️⃣ Decode segmentation mask (RLE format)
            mask = mask_utils.decode(segmentation)
            if mask is None:
                continue  # Skip if mask is empty

            # ✅ 6️⃣ Ensure mask is properly overlaid
            if np.any(mask):  # Only overlay if the mask is not completely empty
                mask_color = np.zeros_like(frame_rgb)
                mask_color[mask == 1] = [255, 0, 0]  # Red mask
                frame_rgb = cv2.addWeighted(frame_rgb, 0.7, mask_color, 0.3, 0)

    # Convert frame back to BGR for OpenCV display and saving
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Show frame
    cv2.imshow("Annotated Video", frame_bgr)

    # Write frame to output video
    out.write(frame_bgr)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1  # Move to next frame

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Annotated video saved at {output_video_path}")
