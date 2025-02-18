import time
import numpy as np
import torchvision.ops.boxes as box_ops
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from deep_sort_realtime.deepsort_tracker import DeepSort
import decord  # Efficient video decoding
from decord import VideoReader, cpu
import imageio
import cv2  # ✅ For displaying frames
import torch
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
# ✅ Enable fast GPU inference
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ✅ Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

# ✅ Load Detectron2 model (WITHOUT TTA)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Single class (contrail)
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # Your trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confidence threshold
cfg.MODEL.DEVICE = device  # Ensure Detectron2 runs on GPU
cfg.TEST.AUG.FLIP = False  # Disable flipping
cfg.TEST.AUG.MIN_SIZES = (800,)  # Reduce scale augmentations
cfg.TEST.AUG.MAX_SIZE = 1333  # Ensure only one size

predictor = DefaultPredictor(cfg)  # ✅ Removed TTA for faster inference
predictor.model = GeneralizedRCNNWithTTA(cfg, predictor.model)
# ✅ Initialize DeepSORT Tracker
tracker = DeepSort(max_age=50, n_init=3, max_cosine_distance=0.4)

# ✅ Video input/output
input_video_path = "C:\\Users\\joe\\Downloads\\Contrail_dataset.mp4"
output_video_path = "output/tracked_contrails.mp4"

# ✅ Load video with Decord
vr = VideoReader(input_video_path, ctx=cpu(0))
fps = vr.get_avg_fps()
frame_width, frame_height = vr[0].shape[1], vr[0].shape[0]

# ✅ Resize settings (speed optimization)
RESIZE_WIDTH, RESIZE_HEIGHT = 640, 360  # Resize frames before inference

# ✅ Create video writer
writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=None)

for frame_idx in range(len(vr)):
    frame = vr[frame_idx].asnumpy()  # Get frame as NumPy array

    # ✅ Resize frame for faster inference
    frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # ✅ Run Detectron2 inference (on GPU)
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        outputs = predictor(frame_resized)  # ✅ Running without TTA

    torch.cuda.synchronize()
    print(f"Inference time: {time.time() - start_time:.4f} seconds")

    # ✅ Extract masks
    instances = outputs["instances"].to(device)
    if len(instances) == 0:
        writer.append_data(frame)  # Write original frame if no detection
        continue  # Skip frame

    masks = instances.pred_masks  # Shape: [num_objects, H, W]

    # ✅ Merge all masks into one
    fused_mask = masks.any(dim=0).float()  # Combine all masks

    # ✅ Get bounding boxes from merged mask
    fused_mask = fused_mask.unsqueeze(0)  # Add batch dim
    boxes = box_ops.masks_to_boxes(fused_mask)  # Extract bounding boxes

    # ✅ Convert mask to NumPy format for visualization
    mask_np = (fused_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)  # Convert to 8-bit image

    # ✅ Convert mask to BGR color (green overlay)
    mask_colored = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    mask_colored[:, :, 1] = mask_np  # Green channel

    # ✅ Blend mask with original frame
    frame_with_mask = cv2.addWeighted(frame_resized, 1.0, mask_colored, 0.5, 0)

    # ✅ Run DeepSORT tracking
    detections = []
    for bbox in boxes.cpu().numpy():
        x1, y1, x2, y2 = bbox
        detections.append(([x1, y1, x2, y2], 1.0, None))  # 1.0 = dummy confidence

    tracks = tracker.update_tracks(detections, frame=frame_resized)

    # ✅ Draw tracking boxes
    for track in tracks:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id

        # Draw bounding box using OpenCV
        cv2.rectangle(frame_with_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_with_mask, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ✅ Display frame (optional)
    cv2.imshow("Tracking with Masks", frame_with_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ✅ Write frame to output video
    writer.append_data(frame_with_mask)

# ✅ Release resources
writer.close()
cv2.destroyAllWindows()
print(f"✅ Output video saved at: {output_video_path}")
