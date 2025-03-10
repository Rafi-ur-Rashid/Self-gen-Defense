import os
import base64
import requests
from tqdm import tqdm
from util import get_jailbreakV_28k

from openai import OpenAI
import jsonlines

client = OpenAI()

questions = get_jailbreakV_28k('data/JailBreakV_28K.csv')

system_prompt = "You are a helpful assistant."

output = []

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
    
    # print(f"user: {text_prompt}")
    # print(f"Processing: {image_path}")
    response=getResponse(system_prompt, q)
    

    json_file = "out/jailbreakV_gpt4o_responses.jsonl"

    # Writing data incrementally
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "query": q, "response": response})

    # print(f"Assistant: {response}")
