
from util import get_jailbreakV_28k
import jsonlines
from tqdm import tqdm
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse

parser = argparse.ArgumentParser(description="Process a string arguments.")
parser.add_argument("model", type=str, default="anthropic", nargs="?", help="Constitutional AI Model name: anthropic, grok")

args = parser.parse_args()
if "anthropic" in args.model:
    model_name= "HuggingFaceH4/mistral-7b-anthropic"
elif "grok" in args.model:
    model_name= "HuggingFaceH4/mistral-7b-grok"
    
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")

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
    json_file = f"out/jailbreakV_constituional_ai_{args.model}_responses.jsonl"

    # Writing data incrementally
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "query": q, "response": response})


