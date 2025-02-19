import json
import os
from llamaapi import LlamaAPI
from util import extract_jb_seq
from tqdm import tqdm
import pandas as pd
import argparse

meta_api_key = os.environ.get("META_API_KEY")

from openai import OpenAI

client = OpenAI(
    api_key=meta_api_key,
    base_url="https://api.llama-api.com"
)

parser = argparse.ArgumentParser(description="Process two string arguments.")
parser.add_argument("input_file", type=str, nargs="?", help="Input file name (without extension)")
parser.add_argument("attribute", type=str, nargs="?", default="prompt", help="Key name for prompts")

args = parser.parse_args()

system_prompt = "You are a helpful assistant."
filename = args.input_file
questions = extract_jb_seq("data/" + filename + ".json", args.attribute)
Meta_model= "gemma2-27b" #"deepseek-r1" #"llama3.3-70b"  # "llama3.1-70b",
output = []

def getResponse(system_prompt, user_prompt):
    completion = client.chat.completions.create(
        model=Meta_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    if isinstance(completion, list):
        return "No response"
    else:
        return completion.choices[0].message.content

for i in tqdm(range(len(questions))):
    q = questions[i]
    user_prompt = q
    response = getResponse(system_prompt, user_prompt)
    output.append(response)

if len(questions) != len(output):
    raise ValueError("Both lists must have the same length.")

# Create list of dictionaries
data = [{"prompt": p, "response": r} for p, r in zip(questions, output)]

# Save to JSON file
with open("out/" + filename + f"_{Meta_model}_response.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)
