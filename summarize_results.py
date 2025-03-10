import json
import pandas as pd
import numpy as np  # Add numpy for median calculation
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description="Process a string arguments.")
parser.add_argument("file_name", type=str, nargs="?", help="json file name (without extension)")

args = parser.parse_args()

# File path
json_file = f"out/{args.file_name}.json"

# Load JSON data
with open(json_file, 'r') as file:
    data = json.load(file)

# Extract toxicity metrics
metrics = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]

# Initialize dictionary to store sum and count
sums = {metric: 0 for metric in metrics}
values = {metric: [] for metric in metrics}  # To store individual values for median calculation
count = len(data)

# Sum up values for each metric and store individual values for median calculation
for entry in data:
    toxic_scores = entry.get("toxic-bert-score", {})
    for metric in metrics:
        score = toxic_scores.get(metric, 0)
        sums[metric] += score
        values[metric].append(score)

# Compute averages
averages = {metric: sums[metric] / count for metric in metrics}

# Compute medians
medians = {metric: float(np.median(values[metric])) for metric in metrics}  # Using numpy for median

# Print results
print("Averages of toxicity scores:")
print(averages)

print("\nMedians of toxicity scores:")
print(medians)
