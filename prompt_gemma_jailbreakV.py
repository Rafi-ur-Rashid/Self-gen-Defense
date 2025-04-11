import torch
from transformers import pipeline
from tqdm import tqdm
from util import get_jailbreakV_28k

from openai import OpenAI
import jsonlines

questions = get_jailbreakV_28k('data/JailBreakV_28K.csv')

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


for i in tqdm(range(7893, len(questions))):
    q = questions[i]
    
    # print(f"user: {text_prompt}")
    # print(f"Processing: {image_path}")
    response=getResponse(q)
    

    json_file = "out/jailbreakV_gemma_responses.jsonl"

    # Writing data incrementally
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "query": q, "response": response})
