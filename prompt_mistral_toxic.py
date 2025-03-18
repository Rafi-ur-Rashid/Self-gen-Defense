
from util import get_toxic_prompts
from tqdm import tqdm
import json

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load tokenizer and model
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")  # Auto-assign GPU if available

questions= get_toxic_prompts("hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl") 

output = []
# with open('out/Llama3_70B_jbbench_baseline.txt', 'w') as file:
#     pass
    
def getResponse(user_prompt):
    # Create a text generation pipeline
    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate text
    output = text_gen_pipeline(user_prompt, max_length=300, do_sample=True, temperature=1)

    # Print the response
    return output[0]['generated_text']


for i in tqdm(range(len(questions))):
    q = questions[i]
    user_prompt= q #f"Complete the following text in 200 words: {q}"
    print(f"User: {q}")
    # print(user_prompt)
    response=getResponse(user_prompt)
    if q not in response:
        response=q+" "+response
    print(f"Assistant: {response}")
    output.append(response)
    
    # with open('out/gpt_judge_results.txt', 'a') as file:
    #     file.write(response + '\n')
    
if len(questions) != len(output):
    raise ValueError("Both lists must have the same length.")

# Create list of dictionaries
data = [{"prompt": p, "response": r} for p, r in zip(questions, output)] #f"Complete the following text in 200 words: {p}"

# Save to JSON file

with open("out/"+"toxicity_mistral_response2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)


