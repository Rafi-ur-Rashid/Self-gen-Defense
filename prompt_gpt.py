from openai import OpenAI
from util import extract_jb_seq
from tqdm import tqdm
import json

client = OpenAI()

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Process two string arguments.")
parser.add_argument("input_file", type=str, nargs="?", help="Input file name (without extension)")
parser.add_argument("attribute", type=str, nargs="?", default="prompt", help="Key name for prompts")

args = parser.parse_args()

system_prompt = "You are a helpful assistant."
# questions=list(input_df['harmful snippet'])
# input_df = pd.read_csv('out/output_Llama2_qb2_sys_.csv')
filename=args.input_file
questions=extract_jb_seq("data/"+filename+".json", args.attribute)

output = []
# with open('out/Llama3_70B_jbbench_baseline.txt', 'w') as file:
#     pass
    
def getResponse(system_prompt, user_prompt):
  completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "developer", "content": system_prompt},
      {"role": "user", "content": user_prompt}
    ]
  )
  return completion.choices[0].message.content


for i in tqdm(range(len(questions))):
    q = questions[i]
    user_prompt= q
    # print(user_prompt)
    response=getResponse(system_prompt, user_prompt)
    output.append(response)
    
    # with open('out/gpt_judge_results.txt', 'a') as file:
    #     file.write(response + '\n')
    
if len(questions) != len(output):
    raise ValueError("Both lists must have the same length.")

# Create list of dictionaries
data = [{"prompt": p, "response": r} for p, r in zip(questions, output)]

# Save to JSON file

with open("out/"+filename+"_gpt-4o_response.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)