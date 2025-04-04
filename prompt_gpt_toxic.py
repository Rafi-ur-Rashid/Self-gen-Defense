
from openai import OpenAI
from util import get_toxic_prompts
from tqdm import tqdm
import json
import jsonlines
import pandas as pd

client = OpenAI()

import pandas as pd


questions= get_toxic_prompts("hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl", count=100) 

system_prompt = "You are a helpful assistant."
# questions=list(input_df['harmful snippet'])
# input_df = pd.read_csv('out/output_Llama2_qb2_sys_.csv')

output = []
# with open('out/Llama3_70B_jbbench_baseline.txt', 'w') as file:
#     pass
    
def getResponse(system_prompt, user_prompt):
  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {"role": "developer", "content": system_prompt},
      {"role": "user", "content": user_prompt}
    ],
    max_tokens=300
  )
  return completion.choices[0].message.content


for i in tqdm(range(len(questions))):
    q = questions[i]
    user_prompt= f"Complete the following text: ```{q}```"
    # print(f"User: {q}")
    # print(user_prompt)
    response=getResponse(system_prompt, user_prompt)
    # if q not in response:
    #     response=q+" "+response
    # print(f"ChatGPT: {response}")
    output.append(response)
    
    # with open('out/gpt_judge_results.txt', 'a') as file:
    #     file.write(response + '\n')
    json_file="out/toxicity_gpt-4o_response.jsonl"
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "prompt": user_prompt, "response": response})
# if len(questions) != len(output):
#     raise ValueError("Both lists must have the same length.")

# # Create list of dictionaries
# data = [{"prompt": f"Complete the following text in 200 words: \n{p}", "response": r} for p, r in zip(questions, output)]

# # Save to JSON file

# with open("out/"+"toxicity_gpt-4o_response.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=4)