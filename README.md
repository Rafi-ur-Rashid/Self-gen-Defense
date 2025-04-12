# Adversarial Scenario Extrapolation: Improving Language Model Robustness through Self-Generated Defense Strategies


A framework for enhancing language model robustness by generating defensive strategies through adversarial scenario extrapolation.

## 📦 Installation

```bash
pip install -r requirements.txt
```
## 🚀 Usage
## Running Jailbreak Experiments with baseline attacks (Without Our defense)

Dataset: JailbreakV-28K dataset (https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k)

With GPT 4o:
```bash
python prompt_gpt_jailbreakV.py
```
With Meta models (Llama, Deepseek, Gemma):

```bash
python prompt_meta_models_jailbreakV.py <model_name>
```
replace <model_name> with any of the following: llama, deepseek, gemma


## Running Jailbreak Experiments with Our defense

With GPT 4o:
```bash
python self_gen_gpt_jailbreakV.py
```
With Meta models (Llama, Deepseek, Gemma):

```bash
python self_gen_meta_models_jailbreakV.py <model_name>
```
replace <model_name> with any of the following: llama, deepseek, gemma


Running any of the previous four commands will produce a jsonl file containing all the responses from the LLM.

To get GPT-judge results for the LLM responses:

```bash
python gpt_judge_jb.py <file_name>
```
replace <file_name> with the input jsonl file name. 

----------------------------------------------------------------------------

## Running Toxicity Experiments with baseline attacks (Without Our defense)

Dataset: RealToxicityPrompt (https://huggingface.co/datasets/allenai/real-toxicity-prompts)

With GPT 4o:

```bash
python prompt_gpt_toxic.py 
```
With Meta models (Llama, Deepseek, Gemma):

```bash
python prompt_meta_models_toxic.py <model_name>
```
replace <model_name> with any of the following: llama, deepseek, gemma

## Running Toxicity Experiments with Our defense

With GPT 4o:
```bash
python self_gen_gpt_toxic.py
```
With Meta models (Llama, Deepseek, Gemma):

```bash
python self_gen_meta_models_toxic.py <model_name>
```
replace <model_name> with any of the following: llama, deepseek, gemma


Running any of the previous four commands will produce a jsonl file containing all the responses from the LLM

To get Toxicity Score using Toxic BERT:
```bash
python score_toxicity.py <file_name>
```
replace <file_name> with the input jsonl file name

And then get the Score summary by running:
```bash
python summarize_results.py <file_name>
```
replace <file_name> with the input json file name without extension

----------------------------------------------------------------------------


## Running TruthfulQA Experiments with baseline attacks (Without Our defense)

Dataset: TruthfulQA dataset (https://huggingface.co/datasets/truthfulqa/truthful_qa)

With GPT 4o:
```bash
python prompt_gpt_truthful.py
```
With Meta models (Llama, Deepseek, Gemma):

```bash
python prompt_meta_models_truthful.py <model_name>
```
replace <model_name> with any of the following: llama, deepseek, gemma


## Running Jailbreak Experiments with Our defense

With GPT 4o:
```bash
python self_gen_gpt_truthful.py
```
With Meta models (Llama, Deepseek, Gemma):

```bash
python self_gen_meta_models_truthful.py <model_name>
```
replace <model_name> with any of the following: llama, deepseek, gemma


Running any of the previous four commands will produce a jsonl file containing all the responses from the LLM.

To get GPT-judge results for the LLM responses:

```bash
python gpt_judge_truthful.py <file_name>
```
replace <file_name> with the input jsonl file name. 

----------------------------------------------------------------------------

## Running Benchmark Study Experiments with Constitutional AI and our defense

Run Jailbreak experiments on Mistral base model:
```bash
python prompt_mistral_jailbreakV.py 
```

Run Toxicity experiments on Mistral base model:
```bash
python prompt_mistral_toxic.py 
```

Run Jailbreak experiments on Constitutional-AI specialized Mistral Model:
```bash
python constitutional_ai_mistral_jailbreakV.py <constituion name>
```

Run Toxicity experiments on Constitutional-AI specialized Mistral model:
```bash
python constitutional_ai_mistral_toxic.py <constituion_name>
```
Replace <constituion_name> with anthropic or grok

Run Jailbreak experiments on Mistral base model with our defense applied:
```bash
python self_gen_mistral_jailbreakV.py
```

Run Toxicity experiments on Mistral base model with our defense applied:
```bash
python self_gen_mistral_toxic.py 
```
Running any of these previous scripts from Benchmark Study will generate a jsonl file containing all the responses from the LLM.
Then follow the same steps to get GPT-judge results or toxicity scores as described before.

----------------------------------------------------------------------------

## 📚 Datasets
This project utilizes the following benchmark datasets:

JailbreakV-28k - Comprehensive evaluation framework for jailbreaking attacks

RealToxicityPrompt - Dataset for examining toxic content generation

TruthfulQA - Benchmark for measuring truthfulness/hallucination in question answering


## HolisticBias evaluation

### Download Toxicity model

Download the toxicity model from [here](https://drive.google.com/drive/folders/1gR6QO-jFzx33BgBpZlzEqbgfdIgD52cm?usp=sharing). Ensure the path is `saved_models/unbiased/unbiased.pt`.

### Run the bias script

The script `compute_holistic_bias.py` computes the bias for the collected responses. Here is the usage

```
usage: compute_holistic_bias.py [-h] [--group GROUP] [--split SPLIT] [--json JSON]

options:
  -h, --help     show this help message and exit
  --group GROUP  Group to analyze
  --split SPLIT  Split to analyze
  --json JSON    JSON file with responses
```

For example

```
python compute_holistic_bias.py --json holistic_gpt4o_responses.json --split test --group religion
```