from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np
import torch
from your_model_file import YourModelClass

class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = 'model_data/model_final.pth'  # Use PyTorch model

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)

        # Load PyTorch model instead of TensorFlow
        self.encoder = self.load_pytorch_model(encoder_model_filename)

    def load_pytorch_model(self, model_path):
        """Loads the PyTorch model properly by initializing the architecture."""
        model = YourModelClass()  # Initialize model (use the correct class)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load weights
        model.eval()  # Set model to evaluation mode
        return model

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        # Extract features using PyTorch model
        with torch.no_grad():
            features = self.encoder(torch.tensor(bboxes, dtype=torch.float32))

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id].numpy()))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            id = track.track_id
            tracks.append(Track(id, bbox))
        self.tracks = tracks


class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
