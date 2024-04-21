from ultralytics import YOLO

# Load pretrained yolo model
model = YOLO('runs/detect/yolov8n.pt')

# Train
results = model.train(
   data='/home/argus-vision/vision/VisionTrainingGround/LD/datasets/coco8/coco8.yaml',
   name='yolov8n_coco_cutomized_loss',
   plots=True,
   save=True,
   epochs=5
)


