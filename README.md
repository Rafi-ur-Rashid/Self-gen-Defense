# Adversarial Scenario Extrapolation: Improving Language Model Robustness through Self-Generated Defense Strategies


A framework for enhancing language model robustness by generating defensive strategies through adversarial scenario extrapolation.

## 📦 Installation

```bash
pip install -r requirements.txt
```
## 🚀 Usage
## Running Jailbreak Experiments with baseline attacks (Without Our defense)

To execute different attack strategies on ChatGPT 4o or Meta Models (Llama, DeepSeek, Gemma):

### Baseline Attack using JBbench 100 dataset (https://github.com/JailbreakBench/jailbreakbench.github.io)
```bash
python prompt_gpt.py "jbbench" "goal"
```
```bash
python prompt_meta_models.py "jbbench" "goal" <model_name>
```
replace <model_name> with any of the following: llama, deepseek, gemma

### Random Search Attack (arxiv.org/abs/2404.02151)
```bas
python prompt_gpt.py "gpt-4-rs"
```
```bash
python prompt_meta_models.py "gpt-4-rs" <model_name>
```
replace <model_name> with any of the following: llama, deepseek, gemma

### PAIR Attack (arxiv.org/abs/2310.08419)
```bash
python prompt_gpt.py "gpt-4-pair"
```

```bash
python prompt_meta_models.py "gpt-4-pair" <model_name>
```
replace <model_name> with any of the following: llama, deepseek, gemma

### JBC Attack (ww1.jailbreakchat.com)
```bash
python prompt_gpt.py "gpt-4-jbc"
```
### GCG Attack (arxiv.org/abs/2307.15043)
```bash
python prompt_gpt.py "gpt-4-gcg"
```

## Running Jailbreak Experiments with Our defense

### Baseline Attack using JBbench 100 dataset 
```bash
python self_gen_gpt.py "jbbench"
```

### Random Search Attack 
```bash
python self_gen_gpt.py "gpt-4-rs" "prompt"
```
### PAIR Attack 
```bash
python self_gen_gpt.py "gpt-4-pair" "prompt"
```
### JBC Attack 
```bash
python self_gen_gpt.py "gpt-4-jbc" "prompt"
```
### GCG Attack 
```bash
python self_gen_gpt.py "gpt-4-gcg" "prompt"
```
----------------------------------------------------------------------------

## Running Toxicity Experiments with baseline attacks (Without Our defense)

### Baseline using RealToxicityPrompt dataset (https://huggingface.co/datasets/allenai/real-toxicity-prompts)
With GPT 4o:

```bash
python prompt_gpt_toxic.py 
```
With Meta models (Llama, Deepseek, Gemma):

```bash
python prompt_meta_models_toxic.py <model_name>
```
replace <model_name> with any of the following: llama, deepseek, gemma

To get Toxicity Score using Toxic BERT:
```bash
python score_toxicity.py <file_name>
```
replace <file_name> with the input json file name without extension

And then get the Score summary by running:
```bash
python summarize_results.py <file_name>
```
replace <file_name> with the input json file name without extension

## 📚 Datasets
This project utilizes the following benchmark datasets:

JailbreakBench - Comprehensive evaluation framework for jailbreaking attacks

RealToxicityPrompt - Dataset for examining toxic content generation

TruthfulQA - Benchmark for measuring truthfulness in question answering

AdvGLUE - Adversarial version of the GLUE benchmark