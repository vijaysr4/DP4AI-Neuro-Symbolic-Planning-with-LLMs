# Neuro-Symbolic Planning with Large Language Models and Formal Verification

This repository implements a **neuro-symbolic planning approach** for Blocksworld tasks by combining **Large Language Models (LLMs)** with a **formal verifier** (Z3). It is based on the techniques from the paper:

> **Neuro Symbolic Reasoning for Planning: Counterexample Guided Inductive Synthesis Using Large Language Models and Satisfiability Solving**  
> by Jha et al.

## Overview

The main goal is to investigate whether **detailed counterexample feedback** and **dynamic prompt refinement** can help an LLM generate valid Blocksworld plans more efficiently than using a static prompt alone. In this project:

1. **Blocksworld Domain**: We model a simplified world with several blocks and a table, where the agent can perform actions like `pick_up`, `put_down`, `stack`, and `unstack`.
2. **LLM-Based Planning**: The system prompts a Large Language Model (e.g., GPT‑4, GPT‑4O, GPT‑4O‑Mini) to generate a plan.
3. **Formal Verification**: We use [Z3](https://github.com/Z3Prover/z3) to verify whether the plan transitions the Blocksworld from the initial to the goal configuration.
4. **Counterexample Feedback**: If verification fails, the system identifies a failing prefix and returns it to the LLM for plan refinement.

## Project Structure

```
decision_procedure_for_AI/
├── experimental_results/
│   ├── gpt4_experiment_results.csv
│   ├── gpt-4o-mini_experiment_results.csv
│   └── gpt-4o_experiment_results.csv
├── visualization/
│   ├── gpt4_average_iterations.png
│   ├── gpt-4o-mini_average_iterations.png
│   └── gpt-4o_average_iterations.png
├── block_assignment.py
├── domain.py
├── experiment.py
├── main.py
├── model_selector.py
├── requirements.txt
├── utils.py
└── visualization.py
```



- **experimental_results/**: Contains CSV files with results of experiments for different LLMs (GPT-4, GPT-4O, GPT-4O-Mini).
- **visualization/**: Stores PNG images illustrating average iteration counts vs. problem sizes/configurations for different models.
- **block_assignment.py**: Generates initial and goal conditions for Blocksworld tasks.
- **domain.py**: Defines the Blocksworld `State` and constraints for actions (`pick_up`, `put_down`, `stack`, `unstack`).
- **experiment.py**: Script to run multiple experiments, log results, and store them in CSV files.
- **main.py**: Core script that orchestrates LLM plan generation, verification, and iterative refinement.
- **model_selector.py**: Provides an interface to different LLMs (GPT‑4, GPT‑4O, GPT‑4O‑Mini, LLaMA, DeepSeek, etc.) via OpenAI or Hugging Face.
- **requirements.txt**: Lists Python dependencies (e.g., `z3-solver`, `openai`, `transformers`, etc.).
- **utils.py**: Helper functions for plan parsing, partial failing-prefix checks, and plan verification flow.
- **visualization.py**: Contains code to generate or display graphs from the CSV results.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<YourUsername>/decision_procedure_for_AI.git
   cd decision_procedure_for_AI

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

Make sure you have a working Python 3.8+ environment and any necessary developer tools for installing Z3 and Transformers.

3. **Add Your API Keys**

If you plan to use GPT-4 or GPT-4O variants, set your OpenAI API key in an environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

If you use LLaMA or DeepSeek models, set your Hugging Face token similarly:
```bash
export HUGGING_FACE_HF_TOKEN="hf_..."
```

## Usage

### 1. Running a Single Plan Generation

You can run `main.py` with a specified model and number of blocks. For example:
```bash
python main.py gpt4 3
```
This command attempts to generate a plan with GPT‑4 for a 3-block problem, verifying it with Z3 in an iterative loop.

### 2. Running Experiments

Use `experiment.py` to run multiple experiments for different configurations:
```bash
python experiment.py
```
This script logs iteration counts and success rates, storing the results as CSV files in the `experimental_results/` folder.

### 3. Visualizing Results

`visualization.py` (and the corresponding code in `experiment.py`) generates and displays bar charts of average iterations versus problem sizes/configurations. For example:
```bash
python visualization.py
```
The generated plots are stored in the `visualization/` folder.

## Results

- **GPT‑4:** Enhanced feedback typically reduces iteration counts significantly for 3-block problems.
- **GPT‑4O & GPT‑4O-Mini:** Similar improvements are observed; however, GPT‑4O-Mini shows less consistent benefits for smaller problem sizes.
- **LLaMA / DeepSeek:** In these experiments, these models often fail to produce a valid plan for even 3-block problems within 20 iterations, likely due to reduced capacity from truncation or distillation.

## License

This project is distributed under the **MIT License**. Feel free to use, modify, and distribute it.
