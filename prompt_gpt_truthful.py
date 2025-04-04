
from openai import OpenAI
from util import get_truthfulQA
from tqdm import tqdm
import json
import jsonlines


client = OpenAI()


questions, _, _= get_truthfulQA("data/TruthfulQA.csv") 

system_prompt = "You are a helpful assistant."

output = []

def getResponse(system_prompt, user_prompt):
  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {"role": "developer", "content": system_prompt},
      {"role": "user", "content": user_prompt}
    ],
    # max_tokens=300
  )
  return completion.choices[0].message.content


for i in tqdm(range(len(questions))):
    q = questions[i]
    user_prompt= f"Answer briefly: ```{q}```"
    response=getResponse(system_prompt, user_prompt)

    output.append(response)
    
    json_file="out/truthful_gpt-4o_response.jsonl"
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "prompt": user_prompt, "response": response})
