"""
Simple evaluation with TrainTester class
"""

from train_prune import TrainTester
import os
import argparse

parser = argparse.ArgumentParser(description="Evaluates yolo model on validation data")
parser.add_argument("--data_path", default="datasets/17R_dataset_pruning", help="path to dataset")
parser.add_argument("--base_model", default="yolov8m.pt", help="name of model to start training with")
parser.add_argument("--name", default="yolov8m_17R_fullset_noprune2", help="name to save run as")
parser.add_argument("--output_path", default=".", help="path to folder for output files")
args = parser.parse_args()

val_path = os.path.join(args.data_path, "val")
img_path = os.path.join(val_path, "images")
label_path = os.path.join(val_path, "labels")
trained_path = os.path.join("runs/detect/", args.name, "weights/best.pt")

train_tester = TrainTester(args.data_path, args.base_model, args.name, args.output_path)
#train_tester.eval_model(0, trained_path, img_path, label_path)
train_tester.yolo_eval(trained_path)