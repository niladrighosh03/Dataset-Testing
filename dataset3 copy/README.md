# 🧠 Multi-Expert Response Generation Framework

This repository provides a structured framework for evaluating response generation techniques using the `qwen2_5_7b_instruct` model. The system explores multiple expert strategies for generating, aggregating, and routing conversational AI outputs, with performance tracked using ROUGE scores.

---

## 📁 Folder Structure

The project is organized into four primary experimental setups:

### 🔹 `aggregator/`
Combines the outputs of **three selected expert models** to form a consensus response. Ideal for ensemble-based evaluation where multi-perspective synthesis is valuable.

### 🔹 `router/`
Uses a **routing mechanism** to select the **most appropriate expert model dynamically** for a given input and returns its output as the final response.

### 🔹 `allexp/`
Aggregates the responses from **all expert models**. This method captures the full spectrum of reasoning capabilities for comprehensive response generation.

### 🔹 `single/`
Generates responses **directly from a single model instance** (`qwen2_5_7b_instruct`), serving as the baseline for comparison against ensemble or routed strategies.

---

## 📂 Folder File Structure

Each experimental folder (`aggregator`, `router`, `allexp`, `single`) shares a common structure:

| File | Description |
|------|-------------|
| `foldername.ipynb` | Main notebook for generating responses using the defined strategy. |
| `table.ipynb` | Script for compiling results into a structured table format. |
| `table_foldername.csv` | Final results file containing:<ul><li>User query</li><li>Agent response</li><li>Baseline response</li><li>ROUGE scores</li></ul> |
| `foldername_response.jsonl` | Raw generated responses from baselines. |
| `cleaned_output.jsonl` | Post-processed responses ready for analysis or visualization. |

---

## 📊 Evaluation

All outputs are evaluated using **ROUGE scores**, with detailed tabular data available in each `table_foldername.csv` file. These evaluations allow comparison across different aggregation and routing techniques for both quality and relevance.

---

## 🤖 Model Used

All responses are generated using the **`qwen2_5_7b_instruct`** language model, designed for instruction-following tasks with strong multi-turn dialogue capabilities.

---

## 📌 Additional File

- `conversation.csv`: Common conversation prompts used across all strategies for benchmarking.

---

## 🚀 Getting Started

To replicate any experiment:
1. Open the corresponding `<foldername>.ipynb` notebook.
2. Run cells to generate model responses.
3. Execute `table.ipynb` to aggregate metrics and create summary tables.
4. Analyze `table_<foldername>.csv` and `cleaned_output.jsonl` for detailed output review.


