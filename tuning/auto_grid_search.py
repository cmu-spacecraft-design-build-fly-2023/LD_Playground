from ultralytics import YOLO
import itertools
import numpy as np

# Define the parameter grid
degrees_options = [0, 90, 180]
scale_options = [0.1, 0.2, 0.3]
fliplr_options = [0.0, 0.5]
mosaic_options = [0, 1]
perspective_options = [0.0001, 0.001, 0.01]

# Prepare combinations of all parameters
param_combinations = list(itertools.product(degrees_options, scale_options, fliplr_options, mosaic_options, perspective_options))

# Placeholder for best parameters and their performance
best_params = None
best_performance = -np.inf  # Assuming higher performance metric is better, adjust according to your metric

for combination in param_combinations:
    degrees, scale, fliplr, mosaic, perspective = combination
    
    # Initialize and load the model
    model = YOLO('yolov8m.pt')
    
    # Train the model with the current set of parameters
    results = model.train(
       data='datasets/17R_dataset_w91_n100/dataset.yaml',
       name=f'yolov8m_R17_w19_n100_deg{degrees}_scl{scale}_flp{fliplr}_msc{mosaic}_prsp{perspective}',
       degrees=degrees,
       scale=scale,
       fliplr=fliplr,
       imgsz=576,
       mosaic=mosaic,
       perspective=perspective,
       plots=True,
       save=True,
       resume=False,  # Set to False to not resume from the last checkpoint for fair comparison
       epochs=10  # Reduce the number of epochs for quicker iterations
    )
    
    # Evaluate model performance
    performance = results.metrics['precision']  # Example metric, adjust according to the actual results object
    
    # Update best parameters if current performance is better
    if performance > best_performance:
        best_performance = performance
        best_params = combination

print(f"Best Parameters: Degrees: {best_params[0]}, Scale: {best_params[1]}, FlipLR: {best_params[2]}, Mosaic: {best_params[3]}, Perspective: {best_params[4]}")
print(f"Best Performance: {best_performance}")
