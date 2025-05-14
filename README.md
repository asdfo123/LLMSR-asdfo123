# 🧠 LLMSR\@XLLM25 Submission by `asdfo123`

## 🚀 Overview

This repository contains our submission to the [**LLMSR\@XLLM25** Shared Task](https://xllms.github.io/LLMSR/) on structured reasoning with LLMs.
Our approach employs **few-shot in-context learning** (ICL) with the untuned [`Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), enhanced by:

* Carefully selected demonstrations
* A three-turn, multi-turn conversational prompt format
* Lightweight post-processing to ensure valid, structured JSON output

Despite its simplicity, our method achieved **5th place overall** on the public leaderboard, outperforming many fine-tuned or retrieval-augmented pipelines.

---

## 🛠️ Implementation Details

* **Model**:
  [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) (off-the-shelf, no fine-tuning)

* **Method**:
  Prompt-only ICL with stage-wise prompt tuning:

  * 2-shot for **Question Parsing (QP)**
  * 3-shot for **CoT Parsing & Verification (CP)**

* **Format**:
  Multi-turn chat-style prompting:

  * `System`: instruction and output format constraints
  * `User`: problem description and request
  * `Assistant`: structured JSON prediction

* **Post-processing**:
  A Python validator script checks output structure, trims noise, aligns pairs, and resolves minor formatting issues (e.g., commas, brackets).

---

## 📈 How to Run the Evaluation

Since the model is fully off-the-shelf, **no model weights need to be provided**. To reproduce our results:

```bash
./eval.sh
```

This will automatically generate `result.json` containing your model outputs.

The evaluation script runs the pipeline in two stages:

```bash
# Stage 1: Question Parsing (QP)
python llama_infer_2shots.py --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --task qp --test_data ./Public_Test_A.json --save_data result.json --icl
python process.py --task qp

# Stage 2: CoT Parsing & Verification (CP)
python llama_infer_2shots.py --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --task cp --test_data ./result.json --save_data result.json --icl
python process.py --task cp
```

---

## 🔁 Customizing the Test Set

To evaluate on a different test set (e.g., `Public_Test_B.json`):

1. Replace the path in the **first QP command only**.
    - Example change: `Public_Test_A.json` → `Public_Test_B.json`
3. Leave all other commands unchanged.

---

## ❓ Troubleshooting

If you encounter any issues, please feel free to contact us:

📧 Email: `leeasdfo123@gmail.com`

---

## 📄 Citation

