import os
import json
from argparse import ArgumentParser
from openai import OpenAI
from tqdm import tqdm
import time
from vllm import LLM, SamplingParams
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn.parallel import DataParallel

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

# 新增：设置CUDA设备
def setup_cuda_devices():
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
        devices = visible_devices.split(',')
        if len(devices) > 0:
            torch.cuda.set_device(int(devices[0]))
            return [int(d) for d in devices]
    return list(range(torch.cuda.device_count()))

def load_llm(device):
    os.environ['HF_ENDPOINT']='https://hf-mirror.com'
    checkpoint = "bigcode/octocoder"
    # 明确指定设备映射
    if torch.cuda.device_count() > 1:
        # 多GPU支持
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        # 单GPU支持
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            device_map={0: device},  # 明确指定设备
            low_cpu_mem_usage=True
        ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model,tokenizer

def octocoder_answer(prompt,model,tokenizer,device):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = inputs.shape[1]  # 获取提示词的长度
    outputs = model.generate(inputs,max_new_tokens=256)
    new_tokens = outputs[0, input_length:]
    predict = tokenizer.decode(new_tokens,skip_special_tokens=True)
    return predict

def generate_all_answer(args,model,tokenizer,device):
    prompt_list = get_prompt(args)
    answer_list = []
    for prompt in tqdm(prompt_list, desc="generating answer"):
        answer = octocoder_answer(prompt,model,tokenizer,device)
        answer_list.append(answer)
        # 清理不再需要的缓存
        torch.cuda.empty_cache()
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
    # 设置CUDA设备
    device = setup_cuda_devices()
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
    print(f"Using devices: {device}")
    model,tokenizer = load_llm(device)
    answer_list = generate_all_answer(args,model,tokenizer,device)
    save_answer(args,answer_list)
