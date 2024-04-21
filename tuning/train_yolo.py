from ultralytics import YOLO

# Load pretrained yolo model
model = YOLO('runs/detect/yolov8s.pt')

# Train
results = model.train(
        data='/home/argus-vision/vision/VisionTrainingGround/LD/datasets/17R_n1342_top_salient/dataset.yaml',
        imgsz=1216,
        epochs=100,
        batch=4,
        name='17R_n1342_top_salient_mse',
        degrees= 180.0,
        scale= 0.5,
        perspective= 0.00017,
        lr0= 0.00428,
        lrf= 0.00933,
        momentum= 0.91146,
        weight_decay= 0.00042,
        warmup_epochs= 3.34695,
        warmup_momentum= 0.54918,
        box= 6.50223,
        cls= 0.65876,
        dfl= 1.28939,
        hsv_h= 0.00959,
        hsv_s= 0.59062,
        hsv_v= 0.34087,
        translate= 0.14881,
        shear= 0.0,
        flipud= 0.0,
        fliplr= 0.505,
        mosaic= 0.92206,
        mixup= 0.0,
        copy_paste= 0.0)


