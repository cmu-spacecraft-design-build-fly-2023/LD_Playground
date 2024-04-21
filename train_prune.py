from ultralytics import YOLO
import argparse
import numpy as np
import os
import cv2
import glob
import csv

class TrainTester():
    def __init__(self, data_path, model, name, output_path, conf=0.5):
        self.base_model = YOLO(model)
        self.conf_thres = conf
        self.name = name
        self.data_path = data_path
        self.tot_classes = 0
        self.output_path = output_path
        self.all_classes = []
    
    def val(self, trained_path):
        # call val to get yolo metrics
        model = YOLO(trained_path)
        results = model.val()

    def detect(self, img):
        results = self.trained_model(img)
        box_list = []

        # get model results
        for result in results:
            boxes = result.boxes

            # iterate over detected boxes
            for box in boxes:
                cx, cy, w, h = box.xywh[0]
                cls = box.cls[0].item()
                conf = box.conf[0].item()

                if conf > self.conf_thres:
                    box_list.append([int(cx.item()), int(cy.item()), int(w.item()), int(h.item()), cls, conf])

        box_arr = np.array(box_list)

        return box_arr


    def train(self, resume=False):
        yaml_path = os.path.join(self.data_path, 'dataset.yaml')

        # first time training
        if resume == False:
            results = self.base_model.train(
                data=yaml_path,
                imgsz=576,
                epochs=100,
                batch=4,

                # Augmentation Parameters
                degrees=180,
                scale=0.1,
                fliplr=0.0,
                translate=0.0, 
                shear=0.0,
                mosaic = 0,
                perspective = 0.0001,
                hsv_h=0.055,
                name=self.name,
                plots=True,
                save=True
            )
        else:
            # later training iterations (resume trained model)
            results = self.trained_model.train(
                data=yaml_path,
                imgsz=576,
                epochs=50,
                batch=4,

                # Augmentation Parameters
                degrees=180,
                scale=0.1,
                fliplr=0.0,
                translate=0.0, 
                shear=0.0,
                mosaic = 0,
                perspective = 0.0001,
                hsv_h=0.055,
                name=self.name,
                plots=True,
                save=True,
                resume=True
            )
    
    def prune_landmarks(self, iter, class_mse, missed_classes):
        print("Pruning landmark classes...")
        out_file = 'pruned_iter' + str(iter) + '.txt'
        out_path = os.path.join(self.output_path, out_file)
        pruned = []
        classes = [c for c in class_mse.keys()]
        missed = [c for c in missed_classes.keys()]
	
	    # detected + missed classes
        all_classes = list(set(classes + missed))
        self.tot_classes = len(set(self.all_classes))
        print("total classes: ", self.tot_classes)
        
        # sort by descending error
        sorted_class_mse = dict(sorted(class_mse.items(), key=lambda item: item[1], reverse=True))
        prune_amount = int(0.1*len(sorted_class_mse))
	
	    # prune undetected classes and bottom 10% detected classes (highest MSE)
        for i in range(prune_amount):
            pruned.append(sorted_class_mse[i])
        
        not_detected = set(self.all_classes) - set(classes)
        print("num undetected classes: ", len(not_detected))
        
        pruned = pruned + [c for c in not_detected]
        print("pruned len", len(pruned))
        print("pruned: ", pruned)
        pruned = [str(int(c)) for c in pruned]

        with open(out_path, 'w') as file:
            file.writelines(line + '\n' for line in pruned)

        return pruned, self.tot_classes

    def _get_gt(self, label_file, im):
        gt_boxes = []
        gt_centers = []
        height, width, _ = im.shape
        # get ground-truth bounding boxes
        with open(label_file) as file:
            for line in file:
                label = line.split()
                cls = int(label[0])
                self.all_classes.append(cls)
                cx, cy = label[1], label[2]
                gt_centers.append([cx, cy])
                cx, cy, w, h = float(
                label[1])*width, float(label[2])*height, float(label[3])*width, float(label[4])*height

                gt_boxes.append([int(cx), int(cy), w, h, cls])
        
        return np.array(gt_boxes), gt_centers
    
    def _get_mse(self, gt, det, class_mse):
        # get mse of centers between ground-truth and detected boxes of same class
        gt_centers = gt[:, :2]
        det_centers = det[:, :2]
        classes = gt[:, -1]

        mse = ((gt_centers - det_centers)**2).mean(axis=1)
        for i in range(len(mse)):
            cls = classes[i]
            if cls in class_mse.keys():
                class_mse[cls].append(mse[i])
            else:
                class_mse.update({cls:[mse[i]]})
        # average across an image
        avg_mse = np.mean(mse)

        return mse, avg_mse

    def _get_err(self, gt, det, class_err):
        # calculates err as Euclidean distance
        gt_centers = gt[:, :2]
        det_centers = det[:, :2]
        classes = gt[:, -1]

        dists = np.linalg.norm(gt_centers - det_centers, axis=1)

        for i in range(len(dists)):
            cls = classes[i]
            if cls in class_err.keys():
                class_err[cls].append(dists[i])
            else:
                class_err.update({cls:[dists[i]]})
        # average across an image
        avg_err = np.mean(dists)

        return dists, avg_err

    def _match_classes(self, gt, det):
        det_classes = det[:, -2]
        
        if len(gt) == 0:
            return [], [], [], det_classes
        
        gt_classes = gt[:, -1]
	    # get common classes between detected and ground-truth
        common_classes = list(set(det_classes).intersection(gt_classes))
        
        matched_gt = np.array([box for box in gt if box[-1] in common_classes])
        matched_det = np.array([box for box in det if box[-2] in common_classes])
        
        if len(common_classes) > 1:
            matched_gt = matched_gt[matched_gt[:, -1].argsort()]
            matched_det = matched_det[matched_det[:, -2].argsort()]

	    # track extraneous and missed classes
        extra = [c for c in det_classes if c not in gt_classes]
        missed = [c for c in gt_classes if c not in det_classes]

        return matched_gt, matched_det, missed, extra
    
    def _prune_dets(self, dets):
        # check if multiple of one class detected and take one with higher conf
        pruned_dets = []
        classes = set(dets[:, -2])
        for c in classes:
            counts = np.sum(dets[:, -2] == c)
            if counts > 1:
                same_cls_boxes = dets[(dets[:, -2] == c)]
                cls_box = same_cls_boxes[np.argmax(same_cls_boxes[:, -1])]
                pruned_dets.append(list(cls_box))
            else:
                pruned_dets.append(list(dets[(dets[:, -2] == c)][0]))
        
        return np.array(pruned_dets)


    def eval_model(self, iter, trained_path, img_path, label_path):
        self.trained_model = YOLO(trained_path)

        val_mse = []
        val_err = []
        tot_detected = 0
        tot_landmarks = 0
        num_extra = 0
        missed_classes = {}
        extra_classes = {}
        class_mse = {}
        class_err = {}
        # load validation data (image and text file)
        for img_file in glob.glob(img_path + '/*.jpg'):
            img_name = os.path.basename(img_file).split('.')[0]
            print("img: ", img_name)
            label_file = os.path.join(label_path, img_name+'.txt')

            im = cv2.imread(img_file)

	    # get ground-truth landmarks
            gt_boxes, gt_centers = self._get_gt(label_file, im)
            tot_landmarks += len(gt_boxes)
            # get detected landmarks
            detections = self.detect(im)
            
            if len(detections) == 0 and len(gt_boxes) == 0: # no detections to be made
                continue
            elif len(detections) == 0 and len(gt_boxes) > 0: # missed detections
                for c in gt_boxes[:, -1]:
                    if c in missed_classes.keys():
                        missed_classes[int(c)] += 1
                    else:
                        missed_classes[int(c)] = 1
            elif len(detections) > 0 and len(gt_boxes) == 0: # extraneous detections
                for c in detections[:, -2]:
                    if c in extra_classes.keys():
                        extra_classes[int(c)] += 1
                    else:
                        extra_classes[int(c)] = 1
            else: # some detections made
                detections = self._prune_dets(detections)

                matched_gt, matched_det, missed, extra = self._match_classes(gt_boxes, detections)
                print("Detected {} out of {} landmarks".format(len(matched_gt), len(gt_boxes)))
                tot_detected += len(matched_gt)
                num_extra += len(extra)
                for m in missed:
                    if m in missed_classes.keys():
                        missed_classes[int(m)] += 1
                    else:
                        missed_classes[int(m)] = 1
                for e in extra:
                    if e in extra_classes.keys():
                        extra_classes[int(e)] += 1
                    else:
                        extra_classes[int(e)] = 1

                if len(matched_gt) > 0:
                    mse, avg_img_mse = self._get_mse(matched_gt, matched_det, class_mse)
                    err, avg_img_err = self._get_err(matched_gt, matched_det, class_err)
                    print("Avg MSE of boxes in one image: ", avg_img_mse)
                    val_mse.append(avg_img_mse)
                    val_err += list(err)
                
                #print("Extra dets: ", extra)
                #print("Missed dets: ", missed)
        
        # get avg mse error per class
        avg_class_mse = {}
        avg_class_err = {}
        for cls, errs in class_mse.items():
            avg = np.mean(errs)
            avg_class_mse.update({cls:avg})

        print("avg class mse", avg_class_mse)

        for cls, errs in class_err.items():
            avg = np.mean(errs)
            avg_class_err.update({cls:avg})

        print("avg class err", avg_class_err)
        
        # sort missed and extraneous classes (highest to lowest)
        extra_classes = dict(sorted(extra_classes.items(), key=lambda item: item[1], reverse=True))
        missed_classes = dict(sorted(missed_classes.items(), key=lambda item: item[1], reverse=True))

        avg_mse = np.mean(val_mse)
        avg_err = np.mean(val_err)
        print("Average Euclidean distance error on val set: ", avg_err)
        print("Average MSE on val set: ", avg_mse)
        print("Total landmarks detected: {}/{}".format(tot_detected, tot_landmarks))
        print("Extraneous landmarks detected: {}".format(num_extra))
        print("Extraneous classes: ", extra_classes)
        print("Missed classes: ", missed_classes)

        return avg_mse, avg_err, extra_classes, missed_classes, avg_class_mse, avg_class_err
    
    def yolo_eval(self, trained_path):
        self.trained_model = YOLO(trained_path)
        self.trained_model.val()
    

    def prune_labels(self, pruned, label_path):
        # delete pruned class labels from label files
        for file in glob.glob(label_path + '/*.txt'):
            print(file)
            # go through label files and delete rows with class in pruned
            with open(file, "r") as f:
                lines = f.readlines()
            # write only lines of classes to keep
            with open(file, "w") as f:
                for line in lines:
                    label = line.split()
                    if label[0] not in pruned:
                        f.write(line)
        
        # sort missed and extraneous classes (highest to lowest)
        extra_classes = dict(sorted(extra_classes.items(), key=lambda item: item[1], reverse=True))
        missed_classes = dict(sorted(missed_classes.items(), key=lambda item: item[1], reverse=True))

        avg_mse = np.mean(val_mse)
        print("Average MSE on val set: ", avg_mse)
        print("Total landmarks detected: {}/{}".format(tot_detected, tot_landmarks))
        print("Extraneous landmarks detected: {}".format(num_extra))
        print("Extraneous classes: ", extra_classes)
        print("Missed classes: ", missed_classes)

        return avg_mse, extra_classes, missed_classes, avg_class_mse
    
    def yolo_eval(self, trained_path):
        self.trained_model = YOLO(trained_path)
        self.trained_model.val()
    

    def prune_labels(self, pruned, label_path):
        # delete pruned class labels from label files
        for file in glob.glob(label_path + '/*.txt'):
            print(file)
            # go through label files and delete rows with class in pruned
            with open(file, "r") as f:
                lines = f.readlines()
            # write only lines of classes to keep
            with open(file, "w") as f:
                for line in lines:
                    label = line.split()
                    if label[0] not in pruned:
                        f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates yolo model on validation data")
    parser.add_argument("--data_path", default="datasets/17R_dataset_pruning", help="path to dataset")
    parser.add_argument("--base_model", default="yolov8m.pt", help="name of model to start training with")
    parser.add_argument("--name", default="yolov8m_R17_pruning", help="name to save run as")
    parser.add_argument("--output_path", default=".", help="path to folder for output files")
    parser.add_argument("--min_landmarks", default=1000, help="minimum number of landmarks to keep")
    
    #parser.add_argument('--model_path', type=str, default="./models/")
    args = parser.parse_args()

    data_path = args.data_path

    train_label_path = os.path.join(data_path, "train", "labels")
    val_img_path = os.path.join(data_path, "val", "images")
    val_label_path = os.path.join(data_path, "val", "labels")
    trained_path = os.path.join("runs/detect/", args.name, 'weights', 'best.pt')
    
    out_file = 'eval_results.csv'
    out_path = os.path.join(args.output_path, out_file)

    train_tester = TrainTester(args.data_path, args.base_model, args.name, args.output_path)
    remaining = np.inf
    
    iter = 0

    # iterate until remaining landmarks under a threshold (TODO: or until avg MSE low enough)
    while remaining > args.min_landmarks:
        print("Iter: ", iter)

        # call training
        if iter == 0:
            train_tester.train()
        else:
            train_tester.train(resume=True)

        # call eval
        avg_mse, extra_classes, missed_classes, avg_class_mse = train_tester.eval_model(iter, trained_path, val_img_path, val_label_path)

        # prune landmarks
        pruned, tot_classes = train_tester.prune_landmarks(iter, avg_class_mse, missed_classes)

        results_dict = {"Iter":iter, "Average MSE":avg_mse, "Extraneous Classes":extra_classes, "Missed Classes":missed_classes, "Pruned":pruned}
        with open(out_path, 'w') as csvfile:
            # creating a csv dict writer object
            writer = csv.DictWriter(csvfile, fieldnames=results_dict.keys())
        
            # writing headers (field names)
            writer.writeheader()
        
            # writing data rows
            writer.writerow(results_dict)
        
        remaining = tot_classes - len(pruned)
        print("remaining classeS: ", remaining)

        # modify label files
        train_tester.prune_labels(pruned, train_label_path)
        train_tester.prune_labels(pruned, val_label_path)

        iter += 1
