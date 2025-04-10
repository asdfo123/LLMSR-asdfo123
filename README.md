# Submit by asdfo123

## Overview

Our approach leverages a few-shot ICL strategy with the pre-trained Llama3-8B-Instruct model. We have carefully refined the prompts and implemented a multi-turn conversation format to maximize performance.

## Implementation Details

- **Model**: We utilize the standard Llama3-8B-Instruct model without any fine-tuning
- **Method**: Few-shot in-context learning with optimized prompts
- **Format**: Multi-turn conversation structure for improved reasoning

## Evaluation Instructions

Since we use the pre-trained weights directly from Hugging Face, there's no need to provide additional model weights. Simply run the evaluation script to generate results:

```bash
./eval.sh
```

This will automatically produce the `result.json` file with our model's predictions.

Detailed script is as follows:

```
python llama_infer_2shots.py --model_name meta-llama/Llama3-8B-Instruct --task qp --test_data ./Public_Test_A.json --save_data result.json --icl
python process.py --task qp
python llama_infer_2shots.py --model_name meta-llama/Llama3-8B-Instruct --task cp --test_data ./result.json --save_data result.json --icl
python process.py --task cp
```

### Customizing Test Data

To evaluate on different test datasets:

1. Modify only the test file path in the **FIRST** command (QP task)
    - Example change: `Public_Test_A.json` → `Public_Test_B.json`
2. Keep all other parameters identical


## Troubleshooting

If you encounter any issues while running the script, please contact:
📧 leeasdfo123@gmail.com

