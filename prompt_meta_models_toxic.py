import json
import os
from util import get_toxic_prompts
from tqdm import tqdm
import pandas as pd
import argparse
import jsonlines

meta_api_key = os.environ.get("META_API_KEY")

from openai import OpenAI

client = OpenAI(
    api_key=meta_api_key,
    base_url="https://api.llama-api.com"
)


system_prompt = "You are a helpful assistant."

questions=get_toxic_prompts("hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl", count=100) 

parser = argparse.ArgumentParser(description="Process a string arguments.")
parser.add_argument("model", type=str, default="llama", nargs="?", help="Model name: llama, deepseek, gemma")

args = parser.parse_args()
if "llama" in args.model:
    Meta_model= "llama3.3-70b" #"gemma2-27b" #"deepseek-r1" #"llama3.3-70b"  # "llama3.1-70b",
elif "deepseek" in args.model:
    Meta_model= "deepseek-r1" #"gemma2-27b" #"deepseek-r1" #"llama3.3-70b"  # "llama3.1-70b",
elif "gemma" in args.model:
    Meta_model= "gemma2-27b" #"gemma2-27b" #"deepseek-r1" #"llama3.3-70b"  # "llama3.1-70b",

output = []

def getResponse(system_prompt, user_prompt):
    completion = client.chat.completions.create(
        model=Meta_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=300
        
    )
    if isinstance(completion, list):
        return "No response"
    else:
        return completion.choices[0].message.content

for i in tqdm(range(len(questions))):
    q = questions[i]
    user_prompt= f"Complete the following text: ```{q}```"
    # print(f"User: {q}")
    response = getResponse(system_prompt, user_prompt)
    # if q not in response:
    #     response=q+" "+response
    # print(f"{Meta_model}: {response}")
    output.append(response)
    json_file=f"out/toxicity_{Meta_model}_response.jsonl"
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "prompt": user_prompt, "response": response})    
        
# if len(questions) != len(output):
#     raise ValueError("Both lists must have the same length.")

# # Create list of dictionaries
# data = [{"prompt": f"Complete the following text in 200 words: \n{p}", "response": r} for p, r in zip(questions, output)]

# # Save to JSON file
# with open(f"out/toxicity_{Meta_model}_response.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=4)
