import transformers
import torch
from tqdm import tqdm
from util import get_jailbreakV_28k
import jsonlines


questions = get_jailbreakV_28k('data/JailBreakV_28K.csv')

system_prompt = "You are a helpful assistant."

model_id = "meta-llama/Llama-3.3-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def getResponse(user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=400,
    )
    return outputs[0]["generated_text"][-1]

for i in tqdm(range(len(questions))):
    q = questions[i]
    
    # print(f"user: {text_prompt}")
    # print(f"Processing: {image_path}")
    response=getResponse(q)
    

    json_file = "out/jailbreakV_llama_responses.jsonl"

    # Writing data incrementally
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "query": q, "response": response})

