from torch.utils.data import DataLoader
from detectron2.data import build_detection_train_loader

train_loader = build_detection_train_loader(cfg)
for i, batch in enumerate(train_loader):
    print(f"Batch {i} loaded")
    if i == 5: break  # Test a few batches
