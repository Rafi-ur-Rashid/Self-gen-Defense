
from openai import OpenAI
# from util import get_truthfulQA
from tqdm import tqdm
import json
# import jsonlines
import pandas as pd

client = OpenAI()

import pandas as pd
from argparse import ArgumentParser


   
questions = json.load(open(f"data/holistic/social_biases_test_subset.json", "r"))

system_prompt = "You are a helpful assistant."

output = []

def getResponse(user_prompt):
  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {"role": "developer", "content": system_prompt},
      {"role": "user", "content": user_prompt}
    ],
    max_tokens=100
  )
  return completion.choices[0].message.content
count=0
for group_name, group_data in questions.items():
    if 'prompts' in group_data:
        updated_prompts = []
        for prompt in group_data['prompts']:
            count+=1
            user_prompt=f'Complete the following text within 50 words: "{prompt['text']}"'
            # print(f"User: {user_prompt}")
            response=getResponse(user_prompt)
            # print(f"Chatbot: {response}")
            updated_prompts.append({
                    'id': prompt['id'],
                    'text': prompt['text'],
                    'response': response
                })
            group_data['prompts'] = updated_prompts
            if count>=100:
                break

with open("out/holistic_gpt4o_responses.json", 'w') as f:
        json.dump(questions, f, indent=2)
        
# for i in tqdm(range(len(questions))):
#     q = questions[i]
#     user_prompt= f"Answer briefly: ```{q}```"
#     response=getResponse(system_prompt, user_prompt)

#     output.append(response)
    
#     json_file="out/truthful_gpt-4o_response.jsonl"
#     with jsonlines.open(json_file, mode="a") as writer:
#         writer.write({"id": i, "prompt": user_prompt, "response": response})
