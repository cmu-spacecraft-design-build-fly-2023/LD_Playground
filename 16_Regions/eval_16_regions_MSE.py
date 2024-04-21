from train_prune import TrainTester
import os
import argparse
import csv

regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S', '32S', '32T', '33S', '33T', '53S', '52S', '54S', '54T']
#regions = ['17T', '18S']
base_model = "yolov8s.pt"
output_path = "MSE_eval"

os.makedirs(output_path, exist_ok=True)

for region_id in regions:
    # Construct data_path and name for each model based on region ID
    data_path = f"datasets/{region_id}_top_salient"
    name = f"yolov8s_{region_id}_top_salient"
    
    # Construct paths for evaluation
    val_path = os.path.join(data_path, "test")
    img_path = os.path.join(val_path, "images")
    label_path = os.path.join(val_path, "labels")
    trained_path = os.path.join("runs/detect", name, "weights/best.pt")
    
    # Initialize TrainTester for the current model
    train_tester = TrainTester(data_path, base_model, name, output_path)
    
    # Evaluate the current model
    avg_mse, avg_err, extra_classes, missed_classes, avg_class_mse, avg_class_err = train_tester.eval_model(0, trained_path, img_path, label_path)

    filename = f"{output_path}/{name}_results.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Average Euclidean error on test set", avg_err])
        writer.writerow(["Average MSE on test set", avg_mse])
        writer.writerow(["Extraneous classes", str(extra_classes)])
        writer.writerow(["Missed classes", str(missed_classes)])
        writer.writerow(["Class", "Average MSE", "Average Err"])
        for i in range(len(avg_class_mse.keys())):
            class_id = list(avg_class_mse.keys())[i]
            writer.writerow([class_id, avg_class_mse[class_id], avg_class_err[class_id]])

    print(f"Results saved to '{filename}'")