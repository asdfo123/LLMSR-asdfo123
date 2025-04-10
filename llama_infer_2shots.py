import json
from jsonschema import validate
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

SYSTEM_MESSAGE = "You are a helpful, detailed and accurate analytical assistant for solving logical problems and analyzing responses."

INPUT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

QP_TEMPLATE = '''Given the following question, please list all original fragments in the question that can be the condition that is related to solving the problem. The question is:\n\n{question}\n\nOutput the results in the following list format, do not output other information

[
    "condition 1",
    "condition 2",
    ...
]
'''

CP_TEMPLATE = '''Given the following quertion, conditions in question, and chain of thought that aims at solveing the problem of the question, please extract statements not appears in the conditions from the original fragment of chain of thought, and extract the evidence that each statement based on from original fragments of chain of thought, finally verify each statement is true (correct) or false (incorrect) according to the conditions provided by question.\nThe question is:\n\n{question}\n\nThe conditions are:\n\n{conditions}\n\nThe chain of thought is:\n\n{cot}\n\nOutput the results in the following json format, do not output other information, just output the json.\n'''

CP_FORMAT = '''
[
    {
        "statement": "statement 1",
        "evidence": "evidence 1",
        "Verification": "True of False"
    },
    {
        "statement": "statement 2",
        "evidence": "evidence 2",
        "Verification": "True of False"
    },
    ...
]
'''


# 问题解析的2-shot学习模板 (QP_ICL_TEMPLATE)


QP_ICL_TEMPLATE = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful analytical assistant that parses questions to extract relevant conditions for problem solving.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following question, please list all original fragments in the question that can be the condition that is related to solving the problem.

There are 7 candidates hired by Haier: F, G, H, I, W, X and Y.One of them needs to be assigned to the public relations department, three to the production department, and the other three to the sales department.The personnel allocation of these 7 employees must meet the following conditions: (1) H and Y must be allocated in the same department.(2) F and G cannot be allocated in the same department (3) If X is allocated in the sales department, W is allocated in the production department.(4) F must be allocated in the production department.
Which of the following lists is a complete and accurate list that cannot be assigned to the production department?
A.F, I, X
B.G, H, Y
C.I, W
D.G

Output the list of parsing results, do not output other information.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
[
    "One of them needs to be assigned to the public relations department, three to the production department, and the other three to the sales department",
    "H and Y must be allocated in the same department",
    "F and G cannot be allocated in the same department",
    "If X is allocated in the sales department, W is allocated in the production department",
    "F must be allocated in the production department"
]
```
<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following question, please list all original fragments in the question that can be the condition that is related to solving the problem.

In a magic show, from the seven magicians-G.H.K.L.N.P and Q, choose 6 people to play, and the performance is divided into two teams: 1 team and 2 teams.Each team consists of three positions: front, middle, and back.The magicians on the field happen to occupy one position each.The choice and location of the magician must meet the following conditions: (1) If G or H is arranged to play, they must be in the front.(2) If K is scheduled to play, he must be in the middle.(3) If L is scheduled to play, he must be on team 1.(4) Neither P nor K can be in the same team as N.(5) P cannot be in the same team as Q.(6) If H is in team 2, Q is in the middle of team 1.
Which of the following is an acceptable arrangement for the first team's performance?
A.Before: Q, In: L, After: N
B.Before: L, Middle: K, After: Q
C.Before: Q, In: K, After: P
D.Before: G, In: K, After: P

Output the list of parsing results, do not output other information.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
[
    "In a magic show, from the seven magicians-G.H.K.L.N.P and Q, choose 6 people to play, and the performance is divided into two teams: 1 team and 2 teams",
    "Each team consists of three positions: front, middle, and back",
    "The magicians on the field happen to occupy one position each",
    "If G or H is arranged to play, they must be in the front",
    "If K is scheduled to play, he must be in the middle",
    "If L is scheduled to play, he must be on team 1",
    "Neither P nor K can be in the same team as N",
    "P cannot be in the same team as Q",
    "If H is in team 2, Q is in the middle of team 1"
]
```
<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following question, please list all original fragments in the question that can be the condition that is related to solving the problem.

{question}

Output the list of parsing results, do not output other information.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''



# 3-shot
CP_ICL_TEMPLATE = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful analytical assistant that parses responses to logical problems, extracts statements, evidence, and verifies their correctness.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a question, the conditions provided by the question, and response that aims at solving the problem of the question, please parse the response: extract the statements from the original fragment of the response, and extract the evidence that each statement based on also from original fragments of the response, finally verify each statement is true (correct) or false (incorrect) according to the conditions provided by question.

The question is:

In a magic show, from the seven magicians-G.H.K.L.N.P and Q, choose 6 people to play, and the performance is divided into two teams: 1 team and 2 teams.Each team consists of three positions: front, middle, and back.The magicians on the field happen to occupy one position each.The choice and location of the magician must meet the following conditions: (1) If G or H is arranged to play, they must be in the front.(2) If K is scheduled to play, he must be in the middle.(3) If L is scheduled to play, he must be on team 1.(4) Neither P nor K can be in the same team as N.(5) P cannot be in the same team as Q.(6) If H is in team 2, Q is in the middle of team 1.
If G is in team 1 and K is in team 2, which of the following magicians must be behind team 2?
A.L
B.N
C.P
D.Q

The conditions provided by the question is:

[
    "The magicians on the field happen to occupy one position each",
    "If G or H is arranged to play, they must be in the front",
    "If K is scheduled to play, he must be in the middle",
    "If L is scheduled to play, he must be on team 1",
    "Neither P nor K can be in the same team as N",
    "P cannot be in the same team as Q",
    "If H is in team 2, Q is in the middle of team 1",
    "G is in team 1 and K is in team 2"
]

The response to the question is:

Since G is in team 1, and K is in team 2, we know that L must be in team 1 (condition 3). We also know that P cannot be in the same team as N (condition 4), so P must be in team 2. Since H is in the front of team 1 (condition 1), and Q is in the middle of team 1 (condition 6), we can conclude that N must be in the back of team 1. Since L is in team 1, N must be behind L.

Make sure all "statement" and "evidence" fields are concise and in a single line without any line breaks, bullet points, or formatting. Output the parsing results strictly in the JSON format shown in the previous example, do not output other additional information, explanation, or anything else.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
[
    {{
        "statement": "L must be in team 1",
        "evidence": "If L is scheduled to play, he must be on team 1",
        "Verification": "True"
    }},
    {{
        "statement": "P must be in team 2",
        "evidence": "G is in team 1, and K is in team 2, We also know that P cannot be in the same team as N",
        "Verification": "True"
    }},
    {{
        "statement": "N must be in the back of team 1",
        "evidence": "H is in the front of team 1 (condition 1), and Q is in the middle of team 1 (condition 6)",
        "Verification": "False"
    }},
    {{
        "statement": "N must be behind L",
        "evidence": "L is in team 1",
        "Verification": "False"
    }}
]
```
<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a question, the conditions provided by the question, and response that aims at solving the problem of the question, please parse the response: extract the statements from the original fragment of the response, and extract the evidence that each statement based on also from original fragments of the response, finally verify each statement is true (correct) or false (incorrect) according to the conditions provided by question.

The question is:

In a magic show, from the seven magicians-G.H.K.L.N.P and Q, choose 6 people to play, and the performance is divided into two teams: 1 team and 2 teams.Each team consists of three positions: front, middle, and back.The magicians on the field happen to occupy one position each.The choice and location of the magician must meet the following conditions: (1) If G or H is arranged to play, they must be in the front.(2) If K is scheduled to play, he must be in the middle.(3) If L is scheduled to play, he must be on team 1.(4) Neither P nor K can be in the same team as N.(5) P cannot be in the same team as Q.(6) If H is in team 2, Q is in the middle of team 1.
Which of the following is an acceptable arrangement for the first team's performance?
A.Before: Q, In: L, After: N
B.Before: L, Middle: K, After: Q
C.Before: Q, In: K, After: P
D.Before: G, In: K, After: P

The conditions provided by the question is:

[
    "In a magic show, from the seven magicians-G.H.K.L.N.P and Q, choose 6 people to play, and the performance is divided into two teams: 1 team and 2 teams",
    "Each team consists of three positions: front, middle, and back",
    "The magicians on the field happen to occupy one position each",
    "If G or H is arranged to play, they must be in the front",
    "If K is scheduled to play, he must be in the middle",
    "If L is scheduled to play, he must be on team 1",
    "Neither P nor K can be in the same team as N",
    "P cannot be in the same team as Q",
    "If H is in team 2, Q is in the middle of team 1"
]

The response to the question is:

Let's analyze the conditions given:

1. If G or H is arranged to play, they must be in the front.
2. If K is scheduled to play, he must be in the middle.
3. If L is scheduled to play, he must be on team 1.
4. Neither P nor K can be in the same team as N.
5. P cannot be in the same team as Q.
6. If H is in team 2, Q is in the middle of team 1.

We need to find an acceptable arrangement for the first team's performance, which consists of three positions: front, middle, and back. Let's try to find a possible arrangement for the first team's performance. We can start by considering the conditions one by one.

From condition 1, we know that if G or H is arranged to play, they must be in the front. Let's assume G is in the front.

From condition 3, we know that if L is scheduled to play, he must be on team 1. Since L is not in the front, he must be in the middle or back. Let's assume L is in the middle.

From condition 2, we know that if K is scheduled to play, he must be in the middle. Since L is already in the middle, K cannot be in the middle. Therefore, K must be in the back.

From condition 4, we know that neither P nor K can be in the same team as N. Since K is in the back, P cannot be in the back. Therefore, P must be in the front or middle. However, P cannot be in the same team as Q, so P cannot be in the same team as L. Therefore, P must be in the front.

From condition 6, we know that if H is in team 2, Q is in the middle of team 1. Since G is already in the front, H cannot be in team 1. Therefore, H must be in team 2.

Now we have a possible arrangement for the first team's performance:

Front: G, P
Middle: L
Back: K

Please proceed to the next step

Make sure all "statement" and "evidence" fields are concise and in a single line without any line breaks, bullet points, or formatting. Output the parsing results strictly in the JSON format shown in the previous example, do not output other additional information, explanation, or anything else.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
[
    {{
        "statement": "he must be in the middle or back",
        "evidence": "From condition 1, we know that if G or H is arranged to play, they must be in the front. Let's assume G is in the front.\n\nFrom condition 3, we know that if L is scheduled to play, he must be on team 1. Since L is not in the front",
        "Verification": "false"
    }},
    {{
        "statement": "K must be in the back",
        "evidence": "From condition 2, we know that if K is scheduled to play, he must be in the middle. Since L is already in the middle, K cannot be in the middle.",
        "Verification": "false"
    }},
    {{
        "statement": "P cannot be in the back",
        "evidence": "From condition 4, we know that neither P nor K can be in the same team as N. Since K is in the back",
        "Verification": "false"
    }},
    {{
        "statement": "P must be in the front or middle",
        "evidence": "From condition 4, we know that neither P nor K can be in the same team as N. Since K is in the back, P cannot be in the back.",
        "Verification": "false"
    }},
    {{
        "statement": "P cannot be in the same team as L",
        "evidence": "P cannot be in the same team as Q",
        "Verification": "false"
    }},
    {{
        "statement": "P must be in the front",
        "evidence": "From condition 4, we know that neither P nor K can be in the same team as N. Since K is in the back, P cannot be in the back. Therefore, P must be in the front or middle. However, P cannot be in the same team as Q, so P cannot be in the same team as L.",
        "Verification": "false"
    }},
    {{
        "statement": "H must be in team 2",
        "evidence": "From condition 6, we know that if H is in team 2, Q is in the middle of team 1. Since G is already in the front, H cannot be in team 1.",
        "Verification": "false"
    }}
]
```
<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a question, the conditions provided by the question, and response that aims at solving the problem of the question, please parse the response: extract the statements from the original fragment of the response, and extract the evidence that each statement based on also from original fragments of the response, finally verify each statement is true (correct) or false (incorrect) according to the conditions provided by question.

The question is:

A company has a 6-story office building.The company's finance department, planning department, administration department, sales department, human resources department, and research and development department work here, and each department occupies one of the floors.Known? (1) The two floors of the human resources department and the sales department are not adjacent; (2) The finance department is on the next floor of the planning department; (3) The floor of the administrative department is above the planning department, but in the manpower Below the Resources Department.
If the human resources department is not on the top floor, the floors that the R & D department may be on are?
A.3, 4, 6
B.3, 4, 5
C.4, 5
D.5, 6

The conditions provided by the question is:

[
    "A company has a 6-story office building.",
    "The company's finance department, planning department, administration department, sales department, human resources department, and research and development department work here, and each department occupies one of the floors.",
    "The two floors of the human resources department and the sales department are not adjacent.",
    "The finance department is on the next floor of the planning department.",
    "The floor of the administrative department is above the planning department, but in the manpower Below the Resources Department.",
    "The human resources department is not on the top floor."
]

The response to the question is:

Let's analyze the given information. We know that the human resources department and the sales department are not adjacent, so they cannot be on the same floor. We also know that the finance department is on the floor above the planning department, and the administrative department is above the planning department but below the human resources department. This means that the planning department is on a floor that is below the administrative department and the human resources department. Since the human resources department is not on the top floor, the top floor must be occupied by either the research and development department or the sales department. We can eliminate option D because the sales department cannot be on the top floor. We can also eliminate option C because the research and development department cannot be on the floor below the human resources department. This leaves us with options A and B. We can eliminate option B because the research and development department cannot be on the floor below the administrative department. Therefore, the research and development department must be on either floor 3 or floor 4.

Make sure all "statement" and "evidence" fields are concise and in a single line without any line breaks, bullet points, or formatting. Output the parsing results strictly in the JSON format shown in the previous example, do not output other additional information, explanation, or anything else.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
[
    {{
        "statement": "the human resources department and the sales department cannot be on the same floor",
        "evidence": "The two floors of the human resources department and the sales department are not adjacent",
        "Verification": "false"
    }},
    {{
        "statement": "The planning department is on a floor that is below the administrative department and the human resources department.",
        "evidence": "The finance department is on the floor above the planning department, and the administrative department is above the planning department but below the human resources department.",
        "Verification": "true"
    }},
    {{
        "statement": "The top floor must be occupied by either the research and development department or the sales department.",
        "evidence": "The human resources department is not on the top floor",
        "Verification": "true"
    }},
    {{
        "statement": "We can eliminate option D",
        "evidence": "The sales department cannot be on the top floor",
        "Verification": "false"
    }},
    {{
        "statement": "We can also eliminate option C",
        "evidence": "The research and development department cannot be on the floor below the human resources department",
        "Verification": "true"
    }},
    {{
        "statement": "We can eliminate option B",
        "evidence": "the research and development department cannot be on the floor below the administrative department",
        "Verification": "true"
    }},
    {{
        "statement": "the research and development department must be on either floor 3 or floor 4.",
        "evidence": "This leaves us with options A and B. We can eliminate option B because the research and development department cannot be on the floor below the administrative department",
        "Verification": "false"
    }}
]
```
<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a quertion, the conditions provided by the question, and response that aims at solveing the problem of the question, please parse the response. Now analyze the following:

The question is:

{question}

The conditions provided by the question is:

{conditions}

The response to the question is:

{cot}

Make sure all "statement" and "evidence" fields are concise and in a single line without any line breaks, bullet points, or formatting. Output the parsing results strictly in the JSON format shown in the previous example, do not output other additional information, explanation, or anything else.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''


def load_model(args):
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # quantization_config=BitsAndBytesConfig(
                #     load_in_4bit=True,
                #     llm_int8_threshold=6.0,
                #     llm_int8_has_fp16_weight=False,
                #     bnb_4bit_compute_dtype=torch.float16,
                #     bnb_4bit_use_double_quant=True,
                #     bnb_4bit_quant_type='nf4',
                # ),
                # torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True
            )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

def call_llama(model, tokenizer, prompts, max_new_tokens=1024):
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
    # sequences = model.generate(**inputs, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
    sequences = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prefix_length = inputs.input_ids.shape[1]
    sequences = [s[prefix_length:] for s in sequences]
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    return decoded

def question_parsing(args, tokenizer, model):
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    for sample in tqdm(test_data, desc=args.task):
        question = sample['question']
        
        if args.icl:
            # Use the multi-turn ICL template
            prompt = QP_ICL_TEMPLATE.format(question=question)
            result = call_llama(model, tokenizer, [prompt], 2048)[0]
        else:
            # Use the simple template with system message
            qp_instruction = QP_TEMPLATE.format(question=question)
            prompt = INPUT_TEMPLATE.format(system=SYSTEM_MESSAGE, user=qp_instruction)
            result = call_llama(model, tokenizer, [prompt], 2048)[0]
            
        sample['question_parsing'] = result

    with open(args.save_data, 'w') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=True)

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

def cot_parsing(args, tokenizer, model):
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    for sample in tqdm(test_data, desc=args.task):
        question = sample['question']
        conditions = json.dumps(sample['question_parsing'], indent=4, ensure_ascii=False)
        cot = sample['cot']
        
        if args.icl:
            # Use the multi-turn ICL template
            prompt = CP_ICL_TEMPLATE.format(question=question, conditions=conditions, cot=cot)
            result = call_llama(model, tokenizer, [prompt], 2048)[0]
        else:
            # Use the simple template with system message
            cp_instruction = CP_TEMPLATE.format(question=question, conditions=conditions, cot=cot) + CP_FORMAT
            prompt = INPUT_TEMPLATE.format(system=SYSTEM_MESSAGE, user=cp_instruction)
            result = call_llama(model, tokenizer, [prompt], 2048)[0]
        
        sample['cot_parsing'] = result
        
    with open(args.save_data, 'w') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=True)
        
def main():
    parser = argparse.ArgumentParser(description="queston and cot parsing")
    parser.add_argument('--model_name', type=str,)
    parser.add_argument('--task', type=str, help='qp (question parsing) or cp (cot parsing)')
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--save_data', type=str)
    parser.add_argument('--icl', action='store_true')
    args = parser.parse_args()

    if args.task == 'qp':
        tokenizer, model = load_model(args)
        question_parsing(args, tokenizer, model)
    elif args.task == 'cp':
        tokenizer, model = load_model(args)
        cot_parsing(args, tokenizer, model)

if __name__ == "__main__":
    main()