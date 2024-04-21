import subprocess
import os

regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S', '32S', '32T', '33S', '33T', '53S', '52S', '54S', '54T']
regions = ['17R']

for region_id in regions: 
    # Construct data_path and name for each model based on region ID
    data_path = f"datasets/{region_id}_top_salient"
    name = f"yolov8l_custom_tuned_{region_id}_top_salient"
    
    # Construct paths for evaluation
    val_path = os.path.join(data_path, "test")
    img_path = os.path.join(val_path, "images")
    label_path = os.path.join(val_path, "labels")
    trained_path = os.path.join("../ultralytics/ultralytics/runs/detect", name, "weights/best.pt")
    best_classes_path = os.path.join("runs/detect", name, "best_classes/")
    px_err_path = f"pxerr_eval/{name}_px_err.csv"

    os.makedirs(best_classes_path, exist_ok=True)
    output_path = "util/err_files/"
    os.makedirs(output_path, exist_ok=True)

    # Define the command you want to execute
    command = "python eval_landmarks.py --model_path " + trained_path + " --im_path " + img_path + " --lab_path " + label_path + " --output_path " + output_path + region_id + "_err.npy --calculate_err --save_err"
    print("Running command:", command)
    # Execute the command in the command line
    subprocess.call(command, shell=True)
    command = "python eval_landmarks.py --err_path " + output_path + region_id + "_err.npy --best_classes --save_best_conf --best_classes_path " + best_classes_path + region_id + "_best_classes.npy --best_conf_path " + best_classes_path + region_id + "_best_conf.npy --pxerr_path " + px_err_path + " --px_threshold 10" 
    print("Running command:", command)
    subprocess.call(command, shell=True)

    # /home/argus-vision/vision/VisionTrainingGround/LD/16_Regions/runs/detect/yolov8s_10S_top_salient/weights
