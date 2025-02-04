import json

def extract_jb_seq(file_path, attribute="prompt"):
    # Load the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract the "goal" field from each element in the "jailbreaks" array
    strings = [jb[attribute] for jb in data["jailbreaks"] if jb[attribute] is not None]
    return strings