from ultralytics import YOLO

# Load pretrained yolo model
model = YOLO('runs/detect/yolov8x_n5000_gsd150/weights/last.pt')

# Train
results = model.train(
   data='datasets/17R_dataset_n5000_gsd150/dataset.yaml',
   name='yolov8x_n5000_gsd150',
   degrees=180,
   scale=0.3,
   fliplr=0.0,
   imgsz=1184,
   mosaic=0,
   batch=1,
   perspective=0.001,
   plots=True,
   save=True,
   resume=True,
   epochs=1000,
   patience=300
)


