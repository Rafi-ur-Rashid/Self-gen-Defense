import json
import pandas as pd

# Load JSON data (assuming it's stored in a file named 'data.json')
import argparse


parser = argparse.ArgumentParser(description="Process a string arguments.")
parser.add_argument("file_name", type=str, nargs="?", help="json file name (without extension)")

args = parser.parse_args()

json_file=f"out/{args.file_name}.json"

with open(json_file, 'r') as file:
    data = json.load(file)

# Extract toxicity metrics
metrics = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]

# Initialize dictionary to store sum and count
sums = {metric: 0 for metric in metrics}
count = len(data)

# Sum up values for each metric
for entry in data:
    toxic_scores = entry.get("toxic-bert-score", {})
    for metric in metrics:
        sums[metric] += toxic_scores.get(metric, 0)

# Compute averages
averages = {metric: sums[metric] / count for metric in metrics}

# Print results
print("Averages of toxicity scores:")
print(averages)
