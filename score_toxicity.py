from transformers import pipeline
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
result = toxicity_classifier("This is a toxic statement.")
print(result)