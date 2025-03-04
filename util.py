import json
import pandas as pd

def extract_jb_seq(file_path, attribute="prompt"):
    # Load the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract the "goal" field from each element in the "jailbreaks" array
    strings = [jb[attribute] for jb in data["jailbreaks"] if jb[attribute] is not None]
    return strings

def get_toxic_prompts(url_, count=100, toxic_level=0.97):
    df = pd.read_json(url_, lines=True)
    questions= [x['text'] for x in (df['prompt']) if x['toxicity'] is not None and x['toxicity']>=toxic_level][:count]
    return questions