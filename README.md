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