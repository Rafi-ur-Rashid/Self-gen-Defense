import json

def extract_jb_seq():

    # Load the JSON file
    file_path = "data/jbbench.json"

    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract the "goal" field from each element in the "jailbreaks" array
    goals = [jb["goal"] for jb in data["jailbreaks"]]
    return goals