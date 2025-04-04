
from util import get_toxic_prompts
from tqdm import tqdm
import json
import jsonlines

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load tokenizer and model
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")  # Auto-assign GPU if available

questions= get_toxic_prompts("hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl") 

output = []
    
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
    
    json_file=f"out/toxicity_mistral_response.jsonl"
    with jsonlines.open(json_file, mode="a") as writer:
        writer.write({"id": i, "prompt": user_prompt, "response": response}) 
    


