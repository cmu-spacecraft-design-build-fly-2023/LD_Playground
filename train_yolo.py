from ultralytics import YOLO

# Load pretrained yolo model
model = YOLO('runs/detect/yolov8s.pt')

# Train
results = model.train(
   data='datasets/17R_n1342_top_salient/dataset.yaml',
   name='yolov8s_R17_n1342_top_salient',
   degrees=180,
   scale=0.3,
   fliplr=0.0,
   imgsz=1152,
   mosaic = 0,
   batch=4,
   perspective = 0.0001,
   plots=True,
   save=True,
   epochs=50
)


