import os
from util import get_jailbreakV_28k
import jsonlines
from tqdm import tqdm
import argparse

meta_api_key = os.environ.get("META_API_KEY")

from openai import OpenAI

client = OpenAI(
    api_key=meta_api_key,
    base_url="https://api.llama-api.com"
)


system_prompt = "You are a helpful assistant."

questions = get_jailbreakV_28k('data/JailBreakV_28K.csv')

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
        max_tokens=400
        
    )
    if isinstance(completion, list):
        return "No response"
    else:
        return completion.choices[0].message.content

for i in tqdm(range(len(questions))):
    q = questions[i]
    # print(f"User: {q}")
    response = getResponse(system_prompt, q)
    # print(f"{Meta_model}: {response}")
    json_file = f"out/jailbreakV_{args.model}_responses.jsonl"

    # Writing data incrementally
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "query": q, "response": response})
