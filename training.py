import os
import json
from detectron2.data.transforms import RandomFlip
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from pycocotools import mask as mask_utils
import cv2
import torch.onnx
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
import numpy as np

# Initialize logger
setup_logger()

# Paths
FRAME_FOLDER = "datasets/"  # Folder containing extracted datasets
ANNOTATION_FOLDER = "coco_output/"  # COCO JSON annotation folder
OUTPUT_DIR = "output/"  # Where trained model will be saved
TORCHSCRIPT_PATH = os.path.join(OUTPUT_DIR, "model_traced.pt")
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all annotation files
annotation_files = [f for f in os.listdir(ANNOTATION_FOLDER) if f.endswith(".json")]

from detectron2.data import detection_utils as utils


def load_coco_dataset():
    dataset_dicts = []

    for annotation_file in annotation_files:
        annotation_path = os.path.join(ANNOTATION_FOLDER, annotation_file)

        with open(annotation_path, "r") as f:
            coco_data = json.load(f)

        for img_info in coco_data["images"]:
            frame_idx = img_info["frame_index"]
            video_name = os.path.splitext(annotation_file)[0].replace("_annotations", "")
            video_name_no_ext = video_name.replace(".mp4", "")
            img_filename = f"{video_name_no_ext}_frame_{frame_idx}.jpg"
            img_path = os.path.join(FRAME_FOLDER, video_name_no_ext, img_filename)

            if not os.path.exists(img_path):
                print(f"⚠️ Warning: Frame {img_path} not found, skipping...")
                continue

            record = {
                "file_name": img_path,
                "image_id": img_info["id"],
                "height": img_info["height"],
                "width": img_info["width"],
                "annotations": [],
            }

            for annotation in coco_data["annotations"]:
                if annotation["image_id"] == img_info["id"]:
                    segmentation = annotation["segmentation"]

                    # Ensure valid segmentation (decode and handle RLE format properly)
                    if isinstance(segmentation, dict):  # RLE format
                        binary_mask = mask_utils.decode(segmentation).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        segmentation = [contour.flatten().tolist() for contour in contours if len(contour) >= 6]


                    # Check if segmentation is valid and has data
                    if not segmentation:
                        continue  # Skip if segmentation is empty

                    # Ensure bbox exists, even if no objects are present
                    bbox = annotation["bbox"]
                    record["annotations"].append({
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": segmentation,  # Use polygon format
                        "category_id": 0  # Default category (change as needed)
                    })

            dataset_dicts.append(record)

    return dataset_dicts


# Register dataset
DatasetCatalog.register("contrail_dataset", load_coco_dataset)
MetadataCatalog.get("contrail_dataset").set(thing_classes=["contrail"])

# Configure model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("contrail_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.01  # A slightly higher learning rate can help convergence
cfg.SOLVER.MAX_ITER = 100000
cfg.SOLVER.STEPS = [1000, 2000]  # Adjust based on dataset size and observation
cfg.SOLVER.GAMMA = 0.1  # Learning rate decay factor
cfg.SOLVER.WARMUP_ITERS = 2000  # Gradual warm-up of learning rate
cfg.OUTPUT_DIR = OUTPUT_DIR
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class: contrails
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Set threshold for inference
cfg.MODEL.MASK_ON = True  # Make sure the model's mask head is active
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Increase to ensure masks are learned
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  # Force segmentation head to detect contrails
cfg.SOLVER.WEIGHT_DECAY = 0.0001  # Regularization term


if __name__ == "__main__":
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Save model to .pb file
    print("✅ Training complete. Exporting to ONNX format...")
    torch.save(trainer.model.state_dict(), os.path.join(OUTPUT_DIR, "model_final.pth"))


    print(f"✅ Model successfully saved as TorchScript")