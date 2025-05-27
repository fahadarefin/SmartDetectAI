import os
import yaml
from ultralytics import YOLO

# === USER SETTINGS ===
root_path = r"E:\SmartDetect\YOLO"
epochs = 100
batch_size = 16
imgsz = 640
use_pretrained = False  # Set to True if you want to fine-tune from 'yolov8n.pt'

# === CREATE roi.yaml ===

yaml_dict = {
    'path': root_path.replace("\\", "/"),  # YOLO likes forward slashes
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': 1,
    'names': ['ROI']
}

yaml_path = os.path.join(root_path, 'roi.yaml')

os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

# Now create the file
with open(yaml_path, 'w') as file:
    yaml.dump(yaml_dict, file)

print(f"[✓] data.yaml created at: {yaml_path}")

# === TRAINING ===
model = YOLO('yolov8n.yaml' if not use_pretrained else 'yolov8n.pt')

model.train(
    data=yaml_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    name='yolo_roi',
    device='cpu'  # Use 'cpu' if you're not on GPU
)

print("[✓] Training complete. Check the 'runs/detect/yolo_roi/' folder for results.")
