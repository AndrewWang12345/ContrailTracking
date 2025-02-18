import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Register COCO-style dataset
def register_my_coco_dataset():
    coco_json_path = "path/to/your/coco_annotations.json"
    image_root = "path/to/images"

    # Register the dataset in Detectron2
    register_coco_instances("my_dataset_train", {}, coco_json_path, image_root)

    # Register Metadata for the dataset
    MetadataCatalog.get("my_dataset_train").set(thing_classes=["class1", "class2"])  # Add your class names
    MetadataCatalog.get("my_dataset_train").set(thing_colors=[[255, 0, 0], [0, 255, 0]])  # Optional color for visualization

register_my_coco_dataset()
