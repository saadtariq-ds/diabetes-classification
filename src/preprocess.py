import os
import sys
import yaml
import pandas as pd

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["preprocess"]

def preprocess_data(input_path, output_path):
    # Read the raw data
    data = pd.read_csv(input_path)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data.to_csv(output_path, index=False, header=True)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data(
        input_path=params["input_dir"], 
        output_path=params["output_dir"]
    )