from detoxify import Detoxify
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

import json
import argparse


model = Detoxify('original')

parser = argparse.ArgumentParser(description="Process a string arguments.")
parser.add_argument("file_name", type=str, nargs="?", help="file name")

args = parser.parse_args()

json_file=f"out/{args.file_name}"

# Load the JSON file
if 'jsonl' in json_file:
  with open(json_file, "r", encoding="utf-8") as file:
    questions = [json.loads(line) for line in file]
else:
  with open(json_file, "r") as file:
        questions = json.load(file)

output = []

for question in questions:

    p= question['prompt']
    q = question['response']
    # if q.startswith(p):
    #      q=q[len(p):]
    outputs = model.predict(q)
    
    response = {key: float(value) for key, value in outputs.items()}
    question['toxic-bert-score']=response
    
    # print(q)
if 'jsonl' in json_file:
  with open(json_file, 'w') as f:
    for item in questions:
        f.write(json.dumps(item) + '\n')
else:
  with open(json_file, 'w') as f:
      json.dump(questions, f, indent=4)