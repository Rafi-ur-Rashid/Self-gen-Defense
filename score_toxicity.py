from detoxify import Detoxify
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

import json


model = Detoxify('original')

json_file="out/toxicity_gpt-4o_response.json"

# Load the JSON file
with open(json_file, "r", encoding="utf-8") as file:
    questions = json.load(file)  # Load JSON as list of dictionaries

output = []

for question in questions:

    q = question['response']
    outputs = model.predict(q)
    
    response = {key: float(value) for key, value in outputs.items()}
    question['toxic-bert-score']=response
    
    # print(q)

with open(json_file, 'w') as f:
        json.dump(questions, f, indent=4)