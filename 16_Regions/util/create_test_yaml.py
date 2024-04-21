import os
import yaml

# Base directory containing the dataset folders
base_dir = "/home/argus-vision/vision/VisionTrainingGround/LD/16_Regions/datasets"
desired_directory = "/home/argus-vision/vision/VisionTrainingGround/LD/16_Regions/util/test_yamls"  # Update this path to your desired directory

# Ensure the desired directory exists
os.makedirs(desired_directory, exist_ok=True)

for region_id in os.listdir(base_dir):
    region_dir = os.path.join(base_dir, region_id)
    if os.path.isdir(region_dir):
        dataset_yaml_path = os.path.join(region_dir, "dataset.yaml")
        if os.path.exists(dataset_yaml_path):
            with open(dataset_yaml_path, 'r') as file:
                dataset_info = yaml.safe_load(file)
                
            # Construct the new content for {region_id}_testset.yaml
            new_content = {
                "train": os.path.join(region_dir, dataset_info['train']),
                "val": os.path.join(region_dir, dataset_info['test']),
                "nc": dataset_info['nc']
            }
            
            # Save to {region_id}_testset.yaml in the desired directory
            new_yaml_path = os.path.join(desired_directory, f"{region_id}_testset.yaml")
            with open(new_yaml_path, 'w') as file:
                yaml.dump(new_content, file, default_flow_style=False)
                
            print(f"Processed and saved: {new_yaml_path}")
        else:
            print(f"dataset.yaml not found in {region_dir}")
    else:
        print(f"Skipping {region_id}, not a directory")
