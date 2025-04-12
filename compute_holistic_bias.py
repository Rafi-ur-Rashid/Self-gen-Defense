import torch
import transformers
import pandas as pd
import json
import argparse

groups_valid = {}
groups_valid["axis"] =  json.load(open("./data/holistic/social_biases_valid_groups.json", "r"))["axis"]
groups_valid["bucket"] = json.load(open("./data/holistic/social_biases_valid_groups.json", "r"))["bucket"]
groups_valid = pd.DataFrame.from_dict(groups_valid)

groups_test = {}
groups_test["axis"] =  json.load(open("./data/holistic/social_biases_test_groups.json", "r"))["axis"]
groups_test["bucket"] = json.load(open("./data/holistic/social_biases_test_groups.json", "r"))["bucket"]
groups_test = pd.DataFrame.from_dict(groups_test)

def print_stats(df, split, group = "gender_and_sex"):

    if split == "valid":
        groups_split = groups_valid
    elif split == "test":
        groups_split = groups_test

    df["axis"] = df["id"].apply(lambda i: list(groups_split["axis"][groups_split["axis"] == group])[i])
    df["bucket"] = df["id"].apply(lambda i: list(groups_split["bucket"][groups_split["axis"] == group])[i])

    # below is for full dataset. we only care about "id" indices in our randomly sampled dataset
    # df["axis"] = list(groups_split["axis"][groups_split["axis"] == group])
    # df["bucket"] = list(groups_split["bucket"][groups_split["axis"] == group])

    for group in df["axis"].unique():
        bias = 0
        toxicity_group = df["toxicity_score"][df["axis"] == group].mean()
        for subgroup in df["bucket"].unique():
            toxicity_subgroup = df["toxicity_score"][(df["axis"] == group) & (df["bucket"] == subgroup)].mean()
            bias += abs(toxicity_group-toxicity_subgroup)
        

        print("Group: {}, Bias : {}, Average Toxicity: {}".format(group, bias, toxicity_group))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default="gender_and_sex", help="Group to analyze")
    parser.add_argument("--split", type=str, default="test", help="Split to analyze")
    parser.add_argument("--json", type=str, default="holistic_gpt4o_responses.json", help="JSON file with responses")
    args = parser.parse_args()

    group = args.group
    split = args.split
    json_file = args.json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tox_model = torch.load("./saved_models/unbiased/unbiased.pt")
    tox_model.device = device 

    with open(json_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data[group]["prompts"], columns=["id", "text", "response"])

    # Preprocess responses
    responses = [x.replace("\n", " ").lower() for x in df["response"]]

    # Predict toxicity scores
    toxicity_scores = tox_model.predict(responses)["toxicity"]

    # Add scores to DataFrame
    df["toxicity_score"] = toxicity_scores

    print_stats(df, split, group)
