# Adversarial Scenario Extrapolation: Improving Language Model Robustness through Self-Generated Defense Strategies


A framework for enhancing language model robustness by generating defensive strategies through adversarial scenario extrapolation.

## 📦 Installation

```bash
pip install -r requirements.txt
```
## 🚀 Usage
### Running Experiments with baseline attacks (Without Our defense)

To execute different attack strategies on ChatGPT 4o:

# Baseline Attack using JBbench 100 dataset (https://github.com/JailbreakBench/jailbreakbench.github.io)
```bash
python prompt_gpt.py "jbbench"
```

# Random Search Attack (arxiv.org/abs/2404.02151)
```bash
python prompt_gpt.py "gpt-4-rs"
```
# PAIR Attack (arxiv.org/abs/2310.08419)
```bash
python prompt_gpt.py "gpt-4-pair"
```
# JBC Attack (ww1.jailbreakchat.com)
```bash
python prompt_gpt.py "gpt-4-jbc"
```
# GCG Attack (arxiv.org/abs/2307.15043)
```bash
python prompt_gpt.py "gpt-4-gcg"
```

### Running Experiments with Our defense

# Baseline Attack using JBbench 100 dataset 
```bash
python self_gen_gpt.py "jbbench"
```

# Random Search Attack 
```bash
python self_gen_gpt.py "gpt-4-rs" "prompt"
```
# PAIR Attack 
```bash
python self_gen_gpt.py "gpt-4-pair" "prompt"
```
# JBC Attack 
```bash
python self_gen_gpt.py "gpt-4-jbc" "prompt"
```
# GCG Attack 
```bash
python self_gen_gpt.py "gpt-4-gcg" "prompt"
```


## 📚 Datasets
This project utilizes the following benchmark datasets:

JailbreakBench - Comprehensive evaluation framework for jailbreaking attacks

TruthfulQA - Benchmark for measuring truthfulness in question answering

AdvGLUE - Adversarial version of the GLUE benchmark

RealToxicityPrompt - Dataset for examining toxic content generation
