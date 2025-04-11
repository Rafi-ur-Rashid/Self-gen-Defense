import torch
from util import get_truthfulQA
from tqdm import tqdm
import pandas as pd
from transformers import pipeline
import jsonlines


system_prompt = "You are a helpful assistant."

questions, _, _= get_truthfulQA("data/TruthfulQA.csv") 


output = []

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-27b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",  # replace with "mps" to run on a Mac device
)

def getResponse(user_prompt):
    messages = [
    {"role": "user", "content": user_prompt},
    ]
    outputs = pipe(messages, max_new_tokens=256)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response

for i in tqdm(range(len(questions))):
    q = questions[i]
    user_prompt= f"Answer briefly: ```{q}```"

    response = getResponse(user_prompt)

    json_file=f"out/truthful_gemma_response.jsonl"
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "prompt": user_prompt, "response": response})
        
