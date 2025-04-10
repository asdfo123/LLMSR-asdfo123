import json
from jsonschema import validate
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import re

# 定义 JSON Schema
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "statement": {"type": "string"},
            "evidence": {"type": "string"},
            "Verification": {"type": "string"}
        },
        "required": ["statement", "evidence", "Verification"],
        "additionalProperties": False
    }
}

# 提取 JSON
def extract_json(data):
    try:
        match = re.search(r"\[.*\]", data, re.DOTALL)
        if match:
            json_data = match.group(0)
            return json.loads(json_data)
        else:
            raise ValueError("No JSON found in the input data.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise

def extract_qp():
    try:
        with open('./result.json', 'r') as f:
            test_data = json.load(f)
            for idx, sample in enumerate(tqdm(test_data)):
                result = sample['question_parsing']
                try:
                    json_data = extract_json(result)
                    sample['question_parsing'] = json_data
                except ValueError as e:
                    print(f"Error parsing JSON at index {idx}: {e}")
                    print(f"Problematic sample: {result}")
                except Exception as e:
                    print(f"Unexpected error at index {idx}: {e}")
                    print(f"Problematic sample: {result}")
    except FileNotFoundError:
        print("Error: The file './output.json' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    with open('./result.json', 'w') as f:
        json.dump(test_data, f, indent=4)

def extract_cp():
    try:
        with open('./result.json', 'r') as f:
            test_data = json.load(f)
            for idx, sample in enumerate(tqdm(test_data)):
                result = sample['cot_parsing']
                try:
                    json_data = extract_json(result)
                    sample['cot_parsing'] = json_data
                    validate(instance=json_data, schema=schema)
                except ValueError as e:
                    print(f"Error parsing JSON at index {idx}: {e}")
                    print(f"Problematic sample: {result}")
                except Exception as e:
                    print(f"Unexpected error at index {idx}: {e}")
                    print(f"Problematic sample: {result}")
    except FileNotFoundError:
        print("Error: The file './output.json' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    with open('./result.json', 'w') as f:
        json.dump(test_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process JSON files.")
    parser.add_argument(
        "--task", 
        choices=["qp", "cp"], 
        required=True, 
        help="Specify the task to run: 'qp' for question parsing or 'cp' for COT parsing."
    )
    args = parser.parse_args()

    if args.task == "qp":
        extract_qp()
    elif args.task == "cp":
        extract_cp()

if __name__ == "__main__":
    main()

    
