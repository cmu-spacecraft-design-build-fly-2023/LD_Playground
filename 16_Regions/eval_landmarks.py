"""
Main evaluation script for calculating metrics and generating visualizations
"""

import os
import numpy as np
from ultralytics import YOLO
import argparse
from tqdm.contrib.concurrent import process_map
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.FloatTensor')  # Sets default tensor type to CPU

def parse_args():
    """
    Parse command line arguments for evaluating landmarks.
    
    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate Landmarks')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--im_path', type=str, default=None, help='Path to the images')
    parser.add_argument('--lab_path', type=str, default=None, help='Path to the labels')
    parser.add_argument('--output_path', type=str, default='err.npy', help='Path to output folder')
    parser.add_argument('--px_threshold', type=float, default=10, help='The maximum pixel mean pixel error for downselection')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='The confidence threshold for detection')
    parser.add_argument('--err_path', type=str, default=None, help='Path to the error file if previously computed')
    parser.add_argument('--calculate_err', action='store_true', help='Calculate the error')
    parser.add_argument('--save_err', action='store_true', help='Save the error')
    parser.add_argument('--best_classes', action='store_true', help='Calculate the best classes')
    parser.add_argument('--save_best_conf', action='store_true', help='Save the best confidence threshold')
    parser.add_argument('--use_best_conf', action='store_true', help='Use the best confidence threshold')
    parser.add_argument('--best_conf_path', type=str, default='best_conf.npy', help='Path to best confidence threshold')
    parser.add_argument('--best_classes_path', type=str, default='best_classes.npy', help='Path to best classes')
    parser.add_argument('--pxerr_path', type=str, default='px_err.csv', help='Path to average px err across best classes')
    parser.add_argument('--evaluate_test', action='store_true', help='Evaluate the test set')
    parsed_args = parser.parse_args()

    if parsed_args.calculate_err and parsed_args.err_path is not None:
        raise ValueError('Cannot calculate error and provide error file at the same time')
    if parsed_args.calculate_err and parsed_args.lab_path is None:
        raise ValueError('Cannot calculate error without labels')
    if parsed_args.calculate_err and parsed_args.im_path is None:
        raise ValueError('Cannot calculate error without images')
    if parsed_args.calculate_err and parsed_args.model_path is None:
        raise ValueError('Cannot calculate error without model')
    
    

    return parsed_args

def load_model(model_path):
    """
    Load the model from the given file path.
    
    Args:
        model_path (str): The path to the model file.
        
    Returns:
        ultralytics.YOLO: The loaded model.
    """
    model = YOLO(model_path).to('cpu')
    return model

def get_err(err_path):
    """
    Load the error file from the given file path.
    
    Args:
        err_path (str): The path to the error file.
        
    Returns:
        numpy.ndarray: The loaded error file.
    """
    err = np.load(err_path)
    return err

def read_labels(lab_path):
    """
    Read the labels from the given file path.
    
    Args:
        lab_path (str): The path to the labels file.
        
    Returns:
        numpy.ndarray: The loaded labels.
    """
    with open(lab_path) as infile:
        labels = infile.readlines()
    if len(labels) > 0:
        labels = [x.split() for x in labels]
        labels = [[int(float(x[0])), float(x[1]), float(x[2])] for x in labels]
    else:
        return []
    return np.array(labels)

def get_dets():
    """
    Get the detections from the given model and image path.
    
    Args:
        model (ultralytics.YOLO): The model to use for inference.
        im_path (str): The path to the images.
        conf_threshold (float): The confidence threshold for detection.
        
    Returns:
        numpy.ndarray: The detections.
    """
    model = load_model(args.model_path)
    results = model(args.im_path, conf=args.conf_threshold)
    dets = []
    print("total result:", len(results))
    for result in results:
        print("Detected landmarks:", len(result.boxes))
        if len(result.boxes) > 0:
            print("     Detected classes:", result.boxes.cls)
            classes = result.boxes.cls
            confs = result.boxes.conf
            xcns = result.boxes.xywhn[:, 0]
            ycns = result.boxes.xywhn[:, 1]
            (im_h, im_w) = result.orig_shape
            im_hs = torch.ones_like(xcns) * im_h
            im_ws = torch.ones_like(ycns) * im_w
            im_dets = torch.stack([classes, xcns, ycns, im_ws, im_hs, confs], dim=1)
            dets.append(im_dets)
            print("     # of classes pulled pixels from:", len(im_dets))
        else:
            dets.append([])
    return dets

def calculate_error(labels, dets, lab_paths):
    """
    Calculate the error between the detections and the labels.
    
    Args:
        labels (list): The labels.
        dets (list): The detections.
        
    Returns:
        numpy.ndarray: The error.
    """
    err_list = []

    for label, det, lab_path in zip(labels, dets, lab_paths):

        if len(det) > 0:
            if len(label) > 0:
                label_classes = label[:, 0]
                label_xcs = label[:, 1]
                label_ycs = label[:, 2]
                for det_cl, det_xc, det_yc, det_im_w, det_im_h, det_conf in det:
                    det_cl = int(det_cl)
                    det_conf = float(det_conf)
                    if det_cl in label_classes:
                        label_idx = np.where(label_classes == det_cl)[0][0]
                        label_x = label_xcs[label_idx]
                        label_y = label_ycs[label_idx]
                        xerr = abs(det_xc - label_x) * det_im_w
                        yerr = abs(det_yc - label_y) * det_im_h
                        err = torch.sqrt(xerr ** 2 + yerr ** 2).cpu().numpy()

                        xerr = (det_xc - label_x) * det_im_w
                        yerr = (det_yc - label_y) * det_im_h

                        err_list.append([det_cl, err, xerr, yerr, det_conf])
                    else:
                        err_list.append([det_cl, -1, 0, 0, det_conf])
            else:
                for det_cl, det_xc, det_yc, det_im_w, det_im_h, det_conf in det:
                    det_cl = int(det_cl)
                    det_conf = float(det_conf) 
                    err_list.append([det_cl, -1, 0, 0, det_conf])
        else:
            if len(label) > 0:
                label_classes = label[:, 0]
                for label_cl, label_x, label_y in label:
                    label_cl = int(label_cl)
                    err_list.append([label_cl, -1, 0, 0, -1])



    err_arr = np.array(err_list)
    
    if args.save_err:
        npy_path = args.output_path
        np.save(npy_path, err_arr)
        print('Error saved to {}'.format(npy_path))

        # Save as a CSV file
        csv_path = args.output_path.replace('.npy', '.csv')
        np.savetxt(csv_path, err_arr, delimiter=',', fmt='%s')
        print('Error saved to {}'.format(csv_path))
    return np.array(err_arr)

def get_class_values(err):
    """
    Get the class values from the error file.
    
    Args:
        err (numpy.ndarray): The error file.
        
    Returns:
        numpy.ndarray: The class values.
    """
    classes = np.unique(err[:, 0])
    return classes

def calculate_class_stats(err, cl, conf_threshold=0.5):
    """
    Calculate the mean error, median error, mean confidence, missed detections, and extra detections for the given class.
    
    Args:
        err (numpy.ndarray): The error file.
        cl (int): The class to calculate the statistics for.
        
    Returns:
        float: The mean error.
        float: The median error.
        float: The mean confidence.
        int: The number of missed detections.
        int: The number of extra detections.
    """
    cl_errs = err[err[:, 0] == cl]
    cl_errs = cl_errs[cl_errs[:, -1] > conf_threshold]
    mean_err = np.nanmean(cl_errs[cl_errs[:,1] > 0, 1])
    median_err = np.nanmedian(cl_errs[cl_errs[:,1] > 0, 1])
    # xy errors
    mean_xerr = np.nanmean(cl_errs[cl_errs[:,1] > 0, 2])
    mean_yerr = np.nanmean(cl_errs[cl_errs[:,1] > 0, 3])
    
    mean_conf = np.nanmean(cl_errs[cl_errs[:,1] > 0, 4])
    missed = np.sum(cl_errs[:, 4] == -1)
    extra = np.sum(cl_errs[:, 1] == -1)
    return cl, mean_err, median_err, mean_xerr, mean_yerr, mean_conf, missed, extra

def get_best_conf(err, min_conf=0.5, max_conf=0.8, steps=20):
    """
    Get the best confidence threshold for the given error file.
    
    Args:
        err (numpy.ndarray): The error file.
        min_conf (float): The minimum confidence threshold.
        max_conf (float): The maximum confidence threshold.
        steps (int): The number of steps to take between the minimum and maximum confidence thresholds.
        
    Returns:
        float: The best confidence threshold.
    """
    confs = np.linspace(min_conf, max_conf, steps)
    best_err = float('inf')
    best_conf = 0
    for conf in confs:
        conf_err = err[err[:, -1] > conf]
        mean_err = np.mean(conf_err[conf_err[:, 1] > 0, 1])
        if mean_err < best_err:
            best_err = mean_err
            best_conf = conf
    return best_conf

def get_best_conf_maximize_classes(err, min_conf=0.5, max_conf=0.90, steps=100):
    """
    Get the best confidence threshold for the given error file by maximizing the number of classes with mean error below the pixel threshold.
    
    Args:
        err (numpy.ndarray): The error file.
        min_conf (float): The minimum confidence threshold.
        max_conf (float): The maximum confidence threshold.
        steps (int): The number of steps to take between the minimum and maximum confidence thresholds.
        
    Returns:
        float: The best confidence threshold.
    """
    confs = np.linspace(min_conf, max_conf, steps)
    best_classes = 0
    best_conf = 0
    for conf in confs:
        conf_err = err[err[:, -1] > conf]
        classes = get_class_values(conf_err)
        class_stats = [calculate_class_stats(conf_err, cl, conf) for cl in classes]
        class_stats = np.array(class_stats)
        class_stats = class_stats[class_stats[:, 0].argsort()]
        choose_classes = class_stats[class_stats[:, 2] < args.px_threshold]
        if len(choose_classes) > best_classes:
            best_classes = len(choose_classes)
            best_conf = conf
            out_classes = choose_classes
    return out_classes, best_conf

def analyze_result_1(best_classes_df, csv_path):
    """
    Calculates the average mean error from the 'Mean Error' column of the best_classes_df DataFrame,
    counts the number of best classes, and saves both pieces of information to a CSV file at the specified path.

    Parameters:
    - best_classes_df: pandas.DataFrame containing the best classes information including 'Mean Error'.
    - csv_path: str, the path where the CSV file containing the average mean error and class count will be saved.
    """

    average_mean_error = best_classes_df['Mean Error'].mean()
    average_mean_x_error = best_classes_df['Mean X Error'].mean()
    average_mean_y_error = best_classes_df['Mean Y Error'].mean()
    num_best_classes = len(best_classes_df)
    summary_df = pd.DataFrame({
        'Average Mean Error': [average_mean_error],
        'Average Mean X Error': [average_mean_x_error],
        'Average Mean Y Error': [average_mean_y_error],
        'Number of Best Classes': [num_best_classes]
    })
    summary_df.to_csv(csv_path, index=False)

    print(f'Average mean error across all classes: {average_mean_error}')
    print(f'Saved the average mean error and class count to {csv_path}')

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(best_classes_df['Mean Error'], bins=len(best_classes_df['Class']), edgecolor='black')
    plt.title('Histogram of Class Mean Error Distribution')
    plt.xlabel('Mean Error')
    plt.ylabel('Frequency')

    # Extract the name from the file_path and create the histogram filename
    directory, filename = os.path.split(csv_path)
    name = filename.replace('_px_err.csv', '')  # Removing the suffix to get the name
    histogram_filename = os.path.join(directory, f'{name}_mean_error_histogram.png')

    # Add text for Average Mean Error and Number of Best Classes on the plot
    text_str = f'Model Name: {name}\nAverage Mean Error: {average_mean_error:.2f}\nNumber of Best Classes: {num_best_classes}'
    plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round", alpha=0.5))
    
    plt.savefig(histogram_filename)
    plt.close()  # Close the figure to avoid displaying it in interactive environments
    print(f'Histogram saved as {histogram_filename}')

    # Second plot: Subplots for Histogram of Class Mean X Error and Mean Y Error Distribution
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Mean X Error subplot
    axs[0].hist(best_classes_df['Mean X Error'], bins=len(best_classes_df), alpha=0.5, label='Mean X Error', edgecolor='black')
    axs[0].axvline(x=0, color='red', linestyle='--')
    axs[0].set_title('Histogram of Class Mean X Error Distribution')
    axs[0].set_xlabel('Mean X Error')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    # Mean Y Error subplot
    axs[1].hist(best_classes_df['Mean Y Error'], bins=len(best_classes_df), alpha=0.5, label='Mean Y Error', edgecolor='black')
    axs[1].axvline(x=0, color='red', linestyle='--')
    axs[1].set_title('Histogram of Class Mean Y Error Distribution')
    axs[1].set_xlabel('Mean Y Error')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    # Adjusting x-axis limits to center 0 for both subplots, if necessary
    min_error = min(best_classes_df['Mean X Error'].min(), best_classes_df['Mean Y Error'].min())
    max_error = max(best_classes_df['Mean X Error'].max(), best_classes_df['Mean Y Error'].max())
    xlims = [-max(abs(min_error), abs(max_error)), max(abs(min_error), abs(max_error))]
    axs[0].set_xlim(xlims)
    axs[1].set_xlim(xlims)

    xy_histogram_filename = os.path.join(directory, f'{name}_mean_xy_error_histogram.png')
    plt.tight_layout()  # Adjust layout to not overlap subplots
    plt.savefig(xy_histogram_filename)
    plt.close()
    print(f'Combined histogram of Mean X and Y Error saved as {xy_histogram_filename}')

def analyze_result_2(best_classes_df, csv_path):
    average_mean_error = best_classes_df['Mean Error'].mean()
    average_mean_x_error = best_classes_df['Mean X Error'].mean()
    average_mean_y_error = best_classes_df['Mean Y Error'].mean()
    num_best_classes = len(best_classes_df)
    summary_df = pd.DataFrame({
        'Average Mean Error': [average_mean_error],
        'Average Mean X Error': [average_mean_x_error],
        'Average Mean Y Error': [average_mean_y_error],
        'Number of Best Classes': [num_best_classes]
    })
    summary_df.to_csv(csv_path, index=False)

    print(f'Average mean error across all classes: {average_mean_error}')
    print(f'Saved the average mean error and class count to {csv_path}')

    # Set figure size to 4:3 aspect ratio, and create subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})
    
    # Merge the upper half for Mean Error
    axs[0, 0].axis('off')  # Turn off the 0,0 plot
    axs[0, 1].axis('off')  # Turn off the 0,1 plot
    ax_mean_error = fig.add_subplot(2, 1, 1)  # Add Mean Error plot across the top half
    
    ax_mean_error.hist(best_classes_df['Mean Error'], bins=len(best_classes_df['Class']), edgecolor='black', alpha=0.5)
    ax_mean_error.set_title('Histogram of Class Mean Error Distribution')
    ax_mean_error.set_xlabel('Mean Error')
    ax_mean_error.set_ylabel('Frequency')
    ax_mean_error.legend()
    
    # Add summary text in the top right corner of the Plot for Mean Error
    text_str = (
        f'Average Mean Error: {average_mean_error:.2f}\n'
        f'Average Mean X Error: {average_mean_x_error:.2f}\n'
        f'Average Mean Y Error: {average_mean_y_error:.2f}\n'
        f'Number of Best Classes: {num_best_classes}'
    )
    ax_mean_error.text(0.95, 0.95, text_str, transform=ax_mean_error.transAxes, fontsize=10, 
                       verticalalignment='top', horizontalalignment='right', 
                       bbox=dict(boxstyle="round", alpha=0.5))

    # Lower left half for Mean X Error
    ax_mean_x_error = axs[1, 0]
    ax_mean_x_error.hist(best_classes_df['Mean X Error'], bins=len(best_classes_df), alpha=0.5, label='Mean X Error', edgecolor='black')
    ax_mean_x_error.axvline(x=0, color='red', linestyle='--')
    ax_mean_x_error.set_title('Mean X Error Distribution')
    ax_mean_x_error.set_xlabel('Mean X Error')
    ax_mean_x_error.set_ylabel('Frequency')
    ax_mean_x_error.legend()

    # Lower right half for Mean Y Error
    ax_mean_y_error = axs[1, 1]
    ax_mean_y_error.hist(best_classes_df['Mean Y Error'], bins=len(best_classes_df), alpha=0.5, label='Mean Y Error', edgecolor='black')
    ax_mean_y_error.axvline(x=0, color='red', linestyle='--')
    ax_mean_y_error.set_title('Mean Y Error Distribution')
    ax_mean_y_error.set_xlabel('Mean Y Error')
    ax_mean_y_error.set_ylabel('Frequency')
    ax_mean_y_error.legend()

    directory, filename = os.path.split(csv_path)
    name = filename.replace('_px_err.csv', '')
    combined_histogram_filename = os.path.join(directory, f'{name}_combined_error_histogram.png')

    plt.tight_layout()
    plt.savefig(combined_histogram_filename)
    plt.close()
    print(f'Combined histogram of Mean Error, Mean X Error, and Mean Y Error saved as {combined_histogram_filename}')

def analyze_result(best_classes_df, csv_path):
    average_mean_error = best_classes_df['Mean Error'].mean()
    average_mean_x_error = best_classes_df['Mean X Error'].mean()
    average_mean_y_error = best_classes_df['Mean Y Error'].mean()
    num_best_classes = len(best_classes_df)
    summary_df = pd.DataFrame({
        'Average Mean Error': [average_mean_error],
        'Average Mean X Error': [average_mean_x_error],
        'Average Mean Y Error': [average_mean_y_error],
        'Number of Best Classes': [num_best_classes]
    })
    summary_df.to_csv(csv_path, index=False)

    print(f'Average mean error across all classes: {average_mean_error}')
    print(f'Saved the average mean error and class count to {csv_path}')

    directory, filename = os.path.split(csv_path)
    name = filename.replace('_px_err.csv', '')
    combined_histogram_filename = os.path.join(directory, f'{name}_combined_error_histogram.png')

    # Create a figure with a custom layout
    fig = plt.figure(figsize=(8, 6))  # Set figure size to 4:3 aspect ratio

    # Add subplot for the top half for Mean Error
    ax_mean_error = fig.add_subplot(211)  # 2 rows, 1 column, 1st subplot
    
    ax_mean_error.hist(best_classes_df['Mean Error'], bins=len(best_classes_df['Class']), edgecolor='black', alpha=0.5)
    ax_mean_error.set_title('Histogram of Class Mean Error Distribution')
    ax_mean_error.set_xlabel('Mean Error')
    ax_mean_error.set_ylabel('Frequency')
    ax_mean_error.legend()

    # Add summary text in the top right corner of the Plot for Mean Error
    text_str = (
        f'Model Name: {name}\n'
        f'Average Mean Error: {average_mean_error:.2f}\n'
        f'Average Mean X Error: {average_mean_x_error:.2f}\n'
        f'Average Mean Y Error: {average_mean_y_error:.2f}\n'
        f'Number of Best Classes: {num_best_classes}'
    )
    ax_mean_error.text(0.95, 0.95, text_str, transform=ax_mean_error.transAxes, fontsize=10, 
                       verticalalignment='top', horizontalalignment='right', 
                       bbox=dict(boxstyle="round", alpha=0.5))

    # Add subplots for the bottom half for Mean X Error (left) and Mean Y Error (right)
    ax_mean_x_error = fig.add_subplot(223)  # 2 rows, 2 columns, 3rd subplot (bottom left)
    ax_mean_x_error.hist(best_classes_df['Mean X Error'], bins=len(best_classes_df), alpha=0.5, label='Mean X Error', edgecolor='black')
    ax_mean_x_error.axvline(x=0, color='red', linestyle='--')
    ax_mean_x_error.set_title('Mean X Error Distribution')
    ax_mean_x_error.set_xlabel('Mean X Error')
    ax_mean_x_error.set_ylabel('Frequency')
    ax_mean_x_error.legend()

    ax_mean_y_error = fig.add_subplot(224)  # 2 rows, 2 columns, 4th subplot (bottom right)
    ax_mean_y_error.hist(best_classes_df['Mean Y Error'], bins=len(best_classes_df), alpha=0.5, label='Mean Y Error', edgecolor='black')
    ax_mean_y_error.axvline(x=0, color='red', linestyle='--')
    ax_mean_y_error.set_title('Mean Y Error Distribution')
    ax_mean_y_error.set_xlabel('Mean Y Error')
    ax_mean_y_error.set_ylabel('Frequency')
    ax_mean_y_error.legend()

    

    plt.tight_layout()
    plt.savefig(combined_histogram_filename)
    plt.close()
    print(f'Combined histogram of Mean Error, Mean X Error, and Mean Y Error saved as {combined_histogram_filename}')


args = parse_args()

if __name__ == '__main__':
    if args.calculate_err:
        lab_folder = args.lab_path   
        lab_paths = sorted(os.listdir(lab_folder))
        lab_paths = [os.path.join(lab_folder, x) for x in lab_paths]
        labels = [read_labels(lab_path) for lab_path in lab_paths] 
        dets = get_dets()
        err = calculate_error(labels, dets, lab_paths)
    elif(args.err_path is not None):
        err = get_err(args.err_path)
    else:
        print('No error file provided or error calculation requested')
    if args.best_classes:
        best_classes, best_conf = get_best_conf_maximize_classes(err)
        if args.save_best_conf:
            np.save(args.best_conf_path, best_conf)
            np.save(args.best_classes_path, np.unique(best_classes[:, 0])) 
            best_classes_csv_path = args.best_classes_path.replace('.npy', '.csv')
            best_conf_csv_path = args.best_conf_path.replace('.npy', '.csv')
            # Save best_classes as CSV
            best_classes_df = pd.DataFrame(best_classes, columns=['Class', 'Mean Error', 'Median Error', 'Mean X Error', 'Mean Y Error', 'Mean Confidence', 'Missed Detections', 'Extra Detections'])
            best_classes_df.to_csv(best_classes_csv_path, index=False)
            # Save best_conf as a single-value CSV
            best_conf_df = pd.DataFrame([best_conf], columns=['Best Confidence Threshold'])
            best_conf_df.to_csv(best_conf_csv_path, index=False)

            print(args.pxerr_path)
            analyze_result(best_classes_df, args.pxerr_path)

        print('Best confidence threshold:', best_conf)
        print('Classes with mean error below {} pixels:'.format(args.px_threshold))
        print(best_classes)
        print('Number of classes:', len(best_classes))
    if args.evaluate_test:
        print('Evaluating test set')
        best_classes = np.load(args.best_classes_path)
        best_conf = np.load(args.best_conf_path)
        print('Best confidence threshold:', best_conf)
        conf_err = err[err[:, -1] > best_conf]
        class_stats = [calculate_class_stats(conf_err, cl, best_conf) for cl in best_classes]
        class_stats = np.array(class_stats)
        class_stats = class_stats[class_stats[:, 0].argsort()]
        print('Class, Mean Error, Median Error, Mean Confidence, Missed Detections, Extra Detections')
        print(class_stats)
        print('Number of classes:', len(best_classes))
        print('Mean median class error:', np.nanmean(class_stats[:, 2]))