from ultralytics import YOLO
from ray import tune
import os

storage_path = "/home/argus-vision/vision/VisionTrainingGround/LD/tuning/runs/detect"
exp_name = "Test_ray_tune_coco8"

# Define a YOLO model
model = YOLO("yolov8m.pt")

# Define the space for augmentation parameters
augmentation_space = {
    "lr0": tune.uniform(1e-5, 1e-1),
}

# Run Ray Tune on the model
result_grid = model.tune(data="/home/argus-vision/vision/VisionTrainingGround/LD/datasets/coco8/coco8.yaml",
                         space=augmentation_space,
                         epochs=3, 
                         grace_period=1,
                         iterations=2, 
                         use_ray=True,
                         )


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
            f"{result.metrics['metrics/mAP50(B)']}\n"
        )

    results_df = result_grid.get_dataframe()
    f.write(f"Results DataFrame:\n{results_df[['training_iteration', 'metrics/mAP50(B)']]}\n")

    f.write(f"Shortest training time: {results_df['time_total_s'].min()}\n")
    f.write(f"Longest training time: {results_df['time_total_s'].max()}\n")

    best_result_df = result_grid.get_dataframe(
        filter_metric="metrics/mAP50(B)", filter_mode="max"
    )
    f.write(f"Best Results DataFrame:\n{best_result_df[['training_iteration', 'metrics/mAP50(B)']]}\n")


import matplotlib.pyplot as plt

# Define the file path for saving the plot
plot_path = os.path.join(storage_path, f"{exp_name}_plot.png")

plt.figure(figsize=(10, 6))
for i, result in enumerate(result_grid):
    if not result.error:
        plt.plot(result.metrics_dataframe["training_iteration"], result.metrics_dataframe["metrics/mAP50(B)"], label=f"Trial {i}")

plt.xlabel('Training Iterations')
plt.ylabel('metrics/mAP50(B)')
plt.legend()
plt.title('Training Iteration vs. metrics/mAP50(B)')
plt.savefig(plot_path)
plt.close()  # Close the plot to free memory
