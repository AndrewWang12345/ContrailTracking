import cv2
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
import os
import json
import numpy as np
from detectron2.structures import BoxMode

# Paths
VIDEO_PATH = "C:\\Users\\joe\\Downloads\\Contrail_dataset.mp4"
OUTPUT_VIDEO_PATH = "C:\\Users\\joe\\PycharmProjects\\ContrailTracking\\output_video.avi"
MODEL_PATH = "C:\\Users\\joe\\PycharmProjects\\ContrailTracking\\output\\model_final.pth"
ANNOTATIONS_PATH = "C:\\Users\\joe\\PycharmProjects\\ContrailTracking\\annotations.json"

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

# Configure the model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class: contrails
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Lower threshold to detect more contrails
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Ensure it runs on GPU

# Create predictor
predictor = DefaultPredictor(cfg)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30  # Default FPS if unable to fetch

# Video writer with a different codec (XVID or MJPG)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can try XVID codec if MJPG fails
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Initialize the list for annotations
annotations = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    outputs = predictor(frame)

    # Collect annotations (segmentation masks)
    frame_annotations = []
    instances = outputs["instances"].to("cpu")
    print(outputs)
    for i in range(len(instances)):
        instance = instances[i]
        mask = instance.pred_masks.numpy()[0]  # Get the mask as a binary array
        score = instance.scores.numpy()[0]
        class_id = instance.pred_classes.numpy()[0]

        # Convert the mask to a list of coordinates (polygon format)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_polygons = [contour.flatten().tolist() for contour in contours]


        annotation = {
            "mask": mask_polygons,  # List of coordinates of the mask's polygon
            "score": score.tolist(),
            "class_id": class_id
        }
        frame_annotations.append(annotation)

    # Add the frame annotations to the list
    annotations.append({
        "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
        "annotations": frame_annotations
    })

    # Visualize the results (ensure masks are drawn)
    v = Visualizer(frame[:, :, ::-1], instance_mode=detectron2.utils.visualizer.ColorMode.IMAGE)
    v = v.draw_instance_predictions(instances)

    # Save frame
    result_frame = v.get_image()[:, :, ::-1]
    out.write(result_frame)

    # Show frame (optional)
    cv2.imshow("Frame", result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save annotations to JSON file
with open(ANNOTATIONS_PATH, "w") as json_file:
    json.dump(annotations, json_file, indent=4)

print(f"✅ Video saved to {OUTPUT_VIDEO_PATH}")
print(f"✅ Annotations saved to {ANNOTATIONS_PATH}")
