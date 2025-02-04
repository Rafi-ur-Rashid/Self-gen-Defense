# Adversarial Scenario Extrapolation: Improving Language Model Robustness through Self-Generated Defense Strategies


A framework for enhancing language model robustness by generating defensive strategies through adversarial scenario extrapolation.

## 📦 Installation

```bash
pip install -r requirements.txt

🚀 Usage
Running Experiments

To execute different attack strategies on ChatGPT 4o:

# Random Search Attack (arxiv.org/abs/2404.02151)
python prompt_gpt.py "gpt-4-rs"

# PAIR Attack (arxiv.org/abs/2310.08419)
python prompt_gpt.py "gpt-4-pair"

# JBC Attack (ww1.jailbreakchat.com)
python prompt_gpt.py "gpt-4-jbc"

# GCG Attack (arxiv.org/abs/2307.15043)
python prompt_gpt.py "gpt-4-gcg"

📚 Datasets
This project utilizes the following benchmark datasets:

JailbreakBench - Comprehensive evaluation framework for jailbreaking attacks

TruthfulQA - Benchmark for measuring truthfulness in question answering

AdvGLUE - Adversarial version of the GLUE benchmark

RealToxicityPrompt - Dataset for examining toxic content generation