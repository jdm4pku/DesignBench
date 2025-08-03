import os
import json
from argparse import ArgumentParser
from openai import OpenAI
from tqdm import tqdm
import time

def llm_parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data",type=str,default="dataset/sysml/dataset.json")
    parser.add_argument("--reason",type=str,default="direct",choices=["direct","few-shot","cot","rag","grammar"])
    parser.add_argument("--prompt_dir",type=str,default="prompts")
    parser.add_argument("--model_type",type=str,default="general",choices=["general","code"])
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument("--moda",type=str,default="greedy")
    parser.add_argument("--output_dir",type=str,default="result")
    return parser.parse_args()

def get_example():
    with open("example.json", 'r', encoding='utf-8') as f:
        example = json.load(f)
    req = example["req"]
    design = example["design"]
    return req, design

def get_sysml_bnf():
    with open("sysml_bnf.txt", 'r', encoding='utf-8') as f:
        sysml_bnf = f.read()
    return sysml_bnf

def get_prompt(args):
    prompt_list = []
    with open(args.data, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    prompt_template_path = os.path.join(args.prompt_dir, f"{args.reason}.txt")
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    for item in dataset:
        nl = item["nl"]
        if args.reason == "few-shot":
            req, design = get_example()
            prompt = prompt_template.format(requirement=nl,req=req,design=design)
        elif args.reason == "grammar":
            sysml_bnf = get_sysml_bnf()
            prompt = prompt_template.format(requirement=nl,sysml_bnf=sysml_bnf)
        else:
            prompt = prompt_template.format(requirement=nl)
        prompt_list.append(prompt)
    return prompt_list

def gpt_4_1_answer(prompt):
    client = OpenAI(
        api_key="sk-BwTI1iSg83soUQ6u2d1096B8A27848E5B3E4141154Dc592b",  # replace with your key
        base_url="https://api.yesapikey.com/v1",
    )
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-4.1-2025-04-14",
                    temperature=0   
            )
            flag=True
        except Exception as e:
            print(e)
            time.sleep(0.5)
    return response.choices[0].message.content

def generate_all_answer(args):
    prompt_list = get_prompt(args)
    answer_list = []
    for prompt in tqdm(prompt_list, desc="generating answer"):
        answer = gpt_4_1_answer(prompt)
        answer_list.append(answer)
    return answer_list

def save_answer(args,answer_list):
    output_dir = os.path.join(args.output_dir, args.model)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{args.reason}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_list, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
    args = llm_parser_args()
    answer_list = generate_all_answer(args)
    save_answer(args,answer_list)



