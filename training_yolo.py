import yaml
import cv2
import os
import json
from ultralytics import YOLO
from pycocotools import mask as mask_utils
import logging

# Set up logging configuration
logging.basicConfig(filename='training_errors.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Paths
FRAME_FOLDER = "C:\\Users\\joe\\PycharmProjects\\ContrailTracking\\datasets\\"  # Folder containing extracted datasets
ANNOTATION_FOLDER = "C:\\Users\\joe\\PycharmProjects\\ContrailTracking\\coco_output\\"  # COCO JSON annotation folder
OUTPUT_DIR = "C:\\Users\\joe\\PycharmProjects\\ContrailTracking\\"  # Directory to save trained model
MODEL_PB_PATH = os.path.join(OUTPUT_DIR, "model_final.pb")
# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load COCO dataset and prepare for YOLO format
def load_coco_dataset():
    dataset = {
        "train": [],
        "val": []
    }

    annotation_files = [f for f in os.listdir(ANNOTATION_FOLDER) if f.endswith(".json")]

    for annotation_file in annotation_files:
        annotation_path = os.path.join(ANNOTATION_FOLDER, annotation_file)

        with open(annotation_path, "r") as f:
            coco_data = json.load(f)

        for img_info in coco_data["images"]:
            frame_idx = img_info["frame_index"]
            video_name = os.path.splitext(annotation_file)[0].replace("_annotations", "")
            video_name_no_ext = video_name.replace(".mp4", "")

            # Adjust img_path creation to use the correct base folder for datasets
            img_filename = f"{video_name_no_ext}_frame_{frame_idx}.jpg"

            # Correcting the frame folder path
            img_path = os.path.join(FRAME_FOLDER, video_name_no_ext,
                                    img_filename)

            # Ensure that the constructed path matches the expected structure
            # Ensure that the constructed path matches the expected structure
            if not os.path.exists(img_path):
                print(f"⚠️ Warning: Frame {img_path} not found, skipping...")
                continue

            record = {
                "image": img_path,
                "annotations": []
            }

            for annotation in coco_data["annotations"]:
                if annotation["image_id"] == img_info["id"]:
                    segmentation = annotation["segmentation"]

                    # Convert RLE to polygons if necessary
                    if isinstance(segmentation, dict):  # RLE format
                        binary_mask = cv2.threshold(mask_utils.decode(segmentation), 0, 255, cv2.THRESH_BINARY)[1]
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        segmentation = [contour.flatten().tolist() for contour in contours if len(contour) >= 6]

                    if not segmentation:
                        continue  # Skip empty segmentation

                    bbox = annotation["bbox"]
                    record["annotations"].append({
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "category_id": 0  # Single-class contrail detection
                    })

            dataset["train"].append(record)

    return dataset


# Load dataset
dataset = load_coco_dataset()

# Create YAML file for YOLOv8
yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
# with open(yaml_path, "w") as f:
#     yaml_data = {
#         "train": [item['image'] for item in dataset["train"]],
#         "val": [item['image'] for item in dataset["val"]],
#         "nc": 1,  # Number of classes
#         "names": ["contrail"],  # Class names
#     }
#     yaml.dump(yaml_data, f)
#
# print(f"✅ Dataset YAML file created at {yaml_path}")


# Train YOLOv8 segmentation model
try:
    print(yaml_path)
    model = YOLO("yolov8n-seg.pt")  # Load pre-trained YOLOv8 segmentation model
    model.train(
        data=yaml_path,  # Use the YAML file for dataset configuration
        epochs=50,
        imgsz=640,
        batch=8,
        workers=4,
        name="yolov8_segmentation",
        save=True
    )
except Exception as e:
    logging.error(f"Error during training: {e}")
    print("An error occurred. Please check the log file for details.")


# Save model as .pb
# print("✅ Training complete. Exporting model to ONNX and TensorFlow format...")
#
# model.export(format="onnx")  # Export to ONNX first
# onnx_model_path = os.path.join(OUTPUT_DIR, "model.onnx")
#
#
#
# onnx_model = onnx.load(onnx_model_path)
# tf_rep = tf2onnx.convert.from_onnx(onnx_model)
# tf_rep.export_graph(MODEL_PB_PATH)
#
# print(f"✅ Model successfully saved as .pb at {MODEL_PB_PATH}")
