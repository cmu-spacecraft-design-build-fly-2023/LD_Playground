from ultralytics import YOLO
from ray import tune
import os

exp_name = "coco_test_run"

# Define a YOLO model
model = YOLO("yolov8s.pt")

# Define the space for augmentation parameters
augmentation_space = {
    "mse": (5, 10),
}

# Run Ray Tune on the model
result_grid = model.tune(data="/home/argus-vision/vision/VisionTrainingGround/LD/datasets/coco8/coco8.yaml",
                         name=exp_name,
                         degrees=180, 
                         scale=0.3, 
                         fliplr=0, 
                         mosaic=0, 
                         perspective=0.0001, 
                         patience=100,
                         epochs=300,
                         space=augmentation_space,
                         iterations=5,
                         plots=True,
                         save=True 
                         )
