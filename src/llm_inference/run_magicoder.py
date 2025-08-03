import os
import json
from argparse import ArgumentParser
from openai import OpenAI
from tqdm import tqdm
import time
from vllm import LLM, SamplingParams
from transformers import pipeline
import torch

os.environ['HF_ENDPOINT']='https://hf-mirror.com'

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

def load_llm():
    os.environ['HF_ENDPOINT']='https://hf-mirror.com'
    generator = pipeline(
        model="ise-uiuc/Magicoder-S-CL-7B",
        task="text-generation",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return generator

def magicoder_answer(prompt,model):
    magicoder_prompt_template = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

    @@ Instruction
    {instruction}

    @@ Response
    """
    final_prompt = magicoder_prompt_template.format(instruction=prompt)
    predict = model(final_prompt,max_length=1024,num_return_sequences=1,do_sample=False)
    generated_text = predict[0]["generated_text"]
    response_start = generated_text.find("@@ Response\n")
    if response_start != -1:
        # 提取响应部分之后的内容
        answer = generated_text[response_start + len("@@ Response\n"):].strip()
        return answer
    else:
        # 如果找不到响应标记，返回全部生成内容
        return generated_text

def generate_all_answer(args,model):
    prompt_list = get_prompt(args)
    answer_list = []
    for prompt in tqdm(prompt_list, desc="generating answer"):
        answer = magicoder_answer(prompt,model)
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
    model = load_llm()
    answer_list = generate_all_answer(args,model)
    save_answer(args,answer_list)
