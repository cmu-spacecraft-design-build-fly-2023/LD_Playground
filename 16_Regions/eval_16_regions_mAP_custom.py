from ultralytics import YOLO
import os
import argparse
import csv
import torch
import csv

torch.set_default_tensor_type('torch.FloatTensor')  # Sets default tensor type to CPU

regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S']

base_model = "yolov8s.pt"
output_path = "custom_mAP_eval"

os.makedirs(output_path, exist_ok=True)

for region_id in regions:
    # Construct data_path and name for each model based on region ID
    data_path = f"datasets/{region_id}_top_salient"
    name = f"yolov8s_custom_{region_id}_top_salient"
    yaml_path = f"util/test_yamls/{region_id}_top_salient_testset.yaml"
    
    # Construct paths for evaluation
    val_path = os.path.join(data_path, "test")
    img_path = os.path.join(val_path, "images")
    label_path = os.path.join(val_path, "labels")
    trained_path = os.path.join("/home/argus-vision/vision/VisionTrainingGround/LD/ultralytics/ultralytics/runs/detect", name, "weights/best.pt")

    # Load model
    model = YOLO(trained_path).to('cpu')

    # Validate the model
    metrics = model.val(data=yaml_path)  # no arguments needed, dataset and settings remembered

    # Define the path for your CSV file
    csv_file_path = os.path.join(output_path, f"{name}_metrics.csv")

    # Open the CSV file in write mode
    with open(csv_file_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)
        
        # Write the header row
        csvwriter.writerow(['Metric', 'Value'])
        
        # Write the mAP, mAP50, and mAP75 metrics
        csvwriter.writerow(['mAP50', metrics.box.map50])
        csvwriter.writerow(['mAP75', metrics.box.map75])
        csvwriter.writerow(['mAP50-95', metrics.box.map])
        
        # Write the mAPs for each category, if available
        if hasattr(metrics.box, 'maps') and isinstance(metrics.box.maps, list):
            for idx, cat_map in enumerate(metrics.box.maps):
                csvwriter.writerow([f'mAP50-95 Category {idx+1}', cat_map])