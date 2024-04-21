from ultralytics import YOLO
from ray import tune
import os

# 16 regions (change this to what datasets are available)
all_regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S', '32S', '32T', '33S', '33T', '53S', '52S', '54S', '54T']
regions = ['10T', '11R', '12R', '16T', '17T', '18S', '32S', '32T', '33S', '33T', '53S', '52S', '54S', '54T']

for r in regions:
    dataset_name = r + '_top_salient'
    print("Training on Dataset: {}".format(dataset_name))
    yaml_path = os.path.join('/home/argus-vision/vision/VisionTrainingGround/LD/16_Regions/datasets', dataset_name, 'dataset.yaml')
    run_name = 'yolov8m_tuning_' + dataset_name
    print("Saving as run: {}".format(run_name))

    # Define a YOLO model
    model = YOLO("yolov8m.pt")

    # Tuning space for full tuning on 17R
    # Define the space for tuning parameters
    tuning_space = {
        "box": (1.0, 20.0),   # box loss gain
        "mse": (1.0, 20.0),   # mse loss gain
        "cls": (1.0, 20.0),   # cls loss gain
        "dfl": (1.0, 20.0),   # dfl loss gain
        "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "translate": (0.0, 0.9),  # image translation (+/- fraction)
        "scale": (0.0, 0.95),  # image scale (+/- gain)
        "shear": (0.0, 10.0),  # image shear (+/- deg)
        "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": (0.0, 1.0),  # image flip up-down (probability),
        "mosaic": (0.0, 1.0)
    }

    # Localized tuning space for remaining regions
    tuning_space = {
        "box": (4.0, 14.0),   # box loss gain
        "mse": (4.0, 14.0),   # mse loss gain
        "cls": (1.0, 7.0),   # cls loss gain
        "dfl": (1.0, 7.0),   # dfl loss gain
        "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "translate": (0.0, 0.5),  # image translation (+/- fraction)
        "scale": (0.0, 0.95),  # image scale (+/- gain)
        "shear": (0.0, 5.0),  # image shear (+/- deg)
        "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": (0.0, 0.5),  # image flip up-down (probability),
        "mosaic": (0.0, 1.0)
    }

    # Run Tuner on the model
    result_grid = model.tune(data=yaml_path,
                            name=run_name,
                            space=tuning_space,
                            optimizer='AdamW',
                            iterations=30, # previously 50
                            degrees=180,
                            fliplr=0.0,
                            imgsz=1216,
                            #mosaic=0.5,
                            batch=12,
                            plots=True,
                            save=True,
                            epochs=20,
                            use_ray=False
                            )
