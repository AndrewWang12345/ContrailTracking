import supervisely as sly
import os
import json
import numpy as np
import pycocotools.mask as mask_utils
import base64
import zlib
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# ✅ Initialize API
server_address = "https://contrailcast.enterprise.supervisely.com"
api_token = "pq4hpk6RLRYGAvZqR2vYmKIylErwuGHscekunRlfmPFaDrKyGEM7f8eggDGCwla8revXa7Yxy4DQvmBmdcoodcDD6WW13Ipp8lnITK1wSdpH2WWAYcCvD0BBS4OXXFSz"
api = sly.Api(server_address, api_token)

# ✅ Define paths
project_id = 3
output_dir = "./coco_output"
video_dir = "./video"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# ✅ Fetch project metadata
project_info = api.project.get_info_by_id(project_id)
if project_info is None:
    raise ValueError(f"Project with ID {project_id} not found!")

# ✅ Get datasets
dataset_id_to_process = 5
datasets = api.dataset.get_list(project_id)
dataset = next((ds for ds in datasets if ds.id == dataset_id_to_process), None)

if dataset is None:
    raise ValueError(f"Dataset with ID {dataset_id_to_process} not found!")

# ✅ Get all videos in the selected dataset
videos = api.video.get_list(dataset.id)

# ✅ Process each video in the selected dataset
for video in videos:
    video_id = video.id
    video_name = video.name

    # ✅ Download video annotation
    annotation_info = api.video.annotation.download(video_id)

    # Prepare COCO structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []  # You may need to fill this based on your classes
    }

    annotation_id = 1  # Counter for annotations

    # ✅ Process datasets
    if "frames" in annotation_info:
        for frame in annotation_info["frames"]:
            frame_index = frame.get("index")
            if frame_index is None:
                continue  # Skip if no frame index

            # ✅ Unique image ID per frame
            frame_image_id = f"{video_id}_{frame_index}"

            # ✅ Add frame metadata
            coco_data["images"].append({
                "id": frame_image_id,
                "file_name": f"{video_name}_frame_{frame_index}.jpg",
                "width": video.frame_width,
                "height": video.frame_height,
                "frame_index": frame_index
            })

            # ✅ Process objects in frame
            for obj in frame.get("figures", []):
                class_id = obj.get("classId", 1)

                # ✅ Handle bitmap mask
                if "geometry" in obj and "bitmap" in obj["geometry"]:
                    bitmap_data = obj["geometry"]["bitmap"]
                    origin_x, origin_y = bitmap_data["origin"]

                    # ✅ Decode mask
                    if isinstance(bitmap_data["data"], str):  # Base64 encoded
                        binary_data = base64.b64decode(bitmap_data["data"])
                        binary_data = zlib.decompress(binary_data)

                        mask_image = Image.open(BytesIO(binary_data))
                        mask_array = np.array(mask_image)

                        mask_h, mask_w = mask_image.size[1], mask_image.size[0]
                        if len(mask_array.shape) == 3 and mask_array.shape[2] == 4:
                            mask_array = mask_array[:, :, 3]

                        mask_array = mask_array.reshape(mask_h, mask_w)
                    else:  # RLE
                        mask_array = mask_utils.decode(bitmap_data["data"])

                    # ✅ Create full-frame mask
                    full_mask = np.zeros((video.frame_height, video.frame_width), dtype=np.uint8)
                    full_mask[origin_y:origin_y + mask_h, origin_x:origin_x + mask_w] = mask_array

                    # ✅ Calculate bounding box
                    mask_indices = np.where(full_mask > 0)
                    if mask_indices[0].size > 0 and mask_indices[1].size > 0:
                        xmin, ymin = np.min(mask_indices[1]), np.min(mask_indices[0])
                        xmax, ymax = np.max(mask_indices[1]), np.max(mask_indices[0])
                        bbox = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
                    else:
                        print(f"⚠️ Warning: Empty mask detected for frame {frame_image_id}")
                        continue

                    # ✅ Convert full mask to RLE
                    rle_full = mask_utils.encode(np.asfortranarray(full_mask))

                    # Convert RLE to a dictionary
                    rle_full_dict = {
                        'counts': rle_full['counts'].decode('utf-8'),
                        'size': rle_full['size']
                    }

                    # ✅ Create COCO annotation
                    coco_annotation = {
                        "id": annotation_id,
                        "image_id": frame_image_id,
                        "category_id": class_id,
                        "segmentation": rle_full_dict,
                        "bbox": bbox,
                        "bbox_mode": 1,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(coco_annotation)
                    annotation_id += 1
    if len(coco_data["annotations"]) == 0:
        print(f"⏭️ Skipping {video_name} (No annotations found)")
        continue  # Move to the next video

    # ✅ Download and save video
    video_path = os.path.join(video_dir, video_name)
    api.video.download_path(video_id, video_path)
    print(f"🎥 Video saved: {video_path}")
    # ✅ Save COCO JSON
    coco_output_file = os.path.join(output_dir, f"{video_name}_annotations.json")
    with open(coco_output_file, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"✅ COCO annotations saved: {coco_output_file}")
