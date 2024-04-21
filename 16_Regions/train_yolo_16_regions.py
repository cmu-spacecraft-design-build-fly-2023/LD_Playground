from ultralytics import YOLO
import os

# 16 regions (change this to what datasets are available)
regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S', '32S', '32T', '33S', '33T', '53S', '52S', '54S', '54T']
#regions = ['17R', '12R', '16T', '54S', '10S', '10T', '11R', '32S', '33S', '33T', '53S', '52S', '54T', '32T', '18S']
regions = ['17R'] # remaining: 32T, 18S, 17T

for r in regions:
    dataset_name = r + '_top_salient'
    print("Training on Dataset: {}".format(dataset_name))
    yaml_path = os.path.join('datasets', dataset_name, 'dataset.yaml')
    run_name = 'yolov8s_' + dataset_name
    print("Saving as run: {}".format(run_name))

    # restart crashed run
    if r == '17T':
        model = YOLO('runs/detect/yolov8s_17T_top_salient3/weights/last.pt')

        # Train
        results = model.train(
            data=yaml_path,
            name=run_name,
            degrees=180,
            scale=0.5,
            fliplr=0.0,
            imgsz=1216,
            mosaic=0.5,
            batch=12,
            perspective=0.0001,
            plots=True,
            save=True,
            resume=True,
            epochs=300
        )
    else:
        # Load pretrained yolo model
        model = YOLO('runs/detect/yolov8s.pt')

        # Train
        results = model.train(
            data=yaml_path,
            name=run_name,
            degrees=180,
            scale=0.5,
            fliplr=0.0,
            imgsz=1216,
            mosaic=0.5,
            batch=12,
            perspective=0.0001,
            plots=True,
            save=True,
            epochs=300
        )
