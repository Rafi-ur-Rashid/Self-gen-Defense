
from util import get_jailbreakV_28k
import jsonlines
from tqdm import tqdm

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# Load tokenizer and model
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")  # Auto-assign GPU if available

questions = get_jailbreakV_28k('data/JailBreakV_28K.csv')[:8000]

    
def getResponse(user_prompt):
    # Create a text generation pipeline
    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate text
    output = text_gen_pipeline(user_prompt, max_new_tokens=600, truncation=True, do_sample=True, temperature=1)

    # Print the response
    return output[0]['generated_text']


for i in tqdm(range(len(questions))):
    q = questions[i]
    user_prompt= q #f"Complete the following text in 200 words: {q}"
    # print(f"User: {q}")
    # print(user_prompt)
    response=getResponse(user_prompt).replace(q, "")

    # print(f"Assistant: {response}")
    json_file = "out/jailbreakV_mistral_responses.jsonl"

    # Writing data incrementally
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "query": q, "response": response})


