import os
from ultralytics import YOLO
from ray import train, tune
from ray.tune import ResultGrid

storage_path = "/home/argus-vision/vision/VisionTrainingGround/LD/tuning/runs/detect"
exp_name = "tune5/_tune_2024-02-15_14-28-46"

def yolo_trainable(config):
    # Initialize the model with the path to your pretrained weights
    model = YOLO("yolov8m.pt")

    # Extract hyperparameters from the config passed by Ray Tune
    lr0 = config["lr0"]

    # Your model's training logic here
    # For example, adjust learning rate based on `lr0`, set up your data, and call any training functions.
    # This is a placeholder for the actual training process, which you need to fill in based on your model's API.
    result = model.train(
        data="/home/argus-vision/vision/VisionTrainingGround/LD/datasets/17R_dataset/dataset.yaml",
        epochs=400,  # or config.get("epochs", 400) if you want to make it tunable
        degrees=180,
        patience=300,
        lr0=lr0,
        # Include any other parameters you wish to tune or configure statically
    )

    # Report metrics back to Ray Tune
    tune.report(loss=result["loss"], accuracy=result["accuracy"])

model = YOLO("yolov8m.pt")

experiment_path = os.path.join(storage_path, exp_name)
print(f"Loading results from {experiment_path}...")

#restored_tuner = tune.Tuner.restore(experiment_path, trainable=model)
restored_tuner = tune.Tuner.restore(experiment_path, trainable="_tune")
result_grid = restored_tuner.get_results()

print(result_grid)

# Define the file path for saving the printout information
printout_path = os.path.join(storage_path, f"{exp_name}_results.txt")

with open(printout_path, "w") as f:
    for i, result in enumerate(result_grid):
        if result.error:
            f.write(f"Trial #{i} had an error: {result.error}\n")
            continue

        f.write(
            f"Trial #{i} finished successfully with a mean accuracy metric of: "
            f"{result.metrics['mean_accuracy']}\n"
        )

    results_df = result_grid.get_dataframe()
    f.write(f"Results DataFrame:\n{results_df[['training_iteration', 'mean_accuracy']]}\n")

    f.write(f"Shortest training time: {results_df['time_total_s'].min()}\n")
    f.write(f"Longest training time: {results_df['time_total_s'].max()}\n")

    best_result_df = result_grid.get_dataframe(
        filter_metric="mean_accuracy", filter_mode="max"
    )
    f.write(f"Best Results DataFrame:\n{best_result_df[['training_iteration', 'mean_accuracy']]}\n")


import matplotlib.pyplot as plt

# Define the file path for saving the plot
plot_path = os.path.join(storage_path, f"{exp_name}_plot.png")

plt.figure(figsize=(10, 6))
for i, result in enumerate(result_grid):
    if not result.error:
        plt.plot(result.metrics_dataframe["training_iteration"], result.metrics_dataframe["mean_accuracy"], label=f"Trial {i}")

plt.xlabel('Training Iterations')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.title('Training Iteration vs. Mean Accuracy')
plt.savefig(plot_path)
plt.close()  # Close the plot to free memory