from nltk.translate.bleu_score import sentence_bleu,corpus_bleu,SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from bert_score import score
import json
import os
from argparse import ArgumentParser
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
BERT_SCORE_MODEL = "google-bert/bert-base-uncased"

def eval_parser_args():
    parser = ArgumentParser()
    parser.add_argument("--preference_path",type=str,default="dataset/sysml/dataset.json")
    parser.add_argument("--predict_dir", type=str, default="predict")
    parser.add_argument("--reason",type=str,default="direct",choices=["direct","few-shot","cot","grammar"])
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument("--output_dir",type=str,default="result")
    return parser.parse_args()

def get_sentece_bleu_score(candidate,reference):
    reference_tokens = reference.split()
    result_tokens = candidate.split()
    # corpus_score = corpus_bleu([[reference_tokens]], [result_tokens], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method3)
    sentence_score = sentence_bleu([reference_tokens], result_tokens,weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=SmoothingFunction().method3)
    return sentence_score


def get_rough_score(generated_answer, reference_answer):
    generated_answer = ','.join(generated_answer)
    reference_answer = ','.join(reference_answer)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # 注意：rouge_scorer 的顺序是 (reference, generated)
    rouge_scores = scorer.score(reference_answer, generated_answer)
    rouge1_f1 = rouge_scores["rouge1"].fmeasure
    rouge2_f1 = rouge_scores["rouge2"].fmeasure
    rougeL_f1 = rouge_scores["rougeL"].fmeasure
    return rouge1_f1,rouge2_f1,rougeL_f1

def get_menteor_score(candidate,reference):
    return single_meteor_score(reference, candidate)

def get_bertscore(candidate,reference,lang="en",rescale_with_baseline=False,use_idf=False):
    P,R,F1 = score([candidate],[reference],lang=lang,
                  model_type= "bert-base-uncased",
                  rescale_with_baseline=rescale_with_baseline,
                  idf=use_idf,
                  verbose=False)
    return F1.mean().item()



def compute_metrics(args):
    preference_data = json.load(open(args.preference_path, 'r', encoding='utf-8'))
    predict_path = f"{args.predict_dir}/{args.model}/{args.reason}.json"
    predict_data = json.load(open(predict_path, 'r', encoding='utf-8'))
    result = []
    all_references = []
    all_candidates = []
    for i,preference in tqdm(enumerate(preference_data),total=len(preference_data),desc="Computing metrics"):
        preference_answer = preference["design"]
        predict_answer = predict_data[i]
        # 分词
        reference_tokens = preference_answer.split()
        candidate_tokens = predict_answer.split()
        all_references.append([reference_tokens])  # corpus_bleu 要求外层 list
        all_candidates.append(candidate_tokens)
        # 单个系统模型的BLEU
        sentence_bleu_score = get_sentece_bleu_score(predict_answer,preference_answer)
        rouge1_f1,rouge2_f1,rougeL_f1 = get_rough_score(predict_answer,preference_answer)
        bertscore = get_bertscore(predict_answer,preference_answer)
        # meteor_score = get_menteor_score(predict_answer,preference_answer)
        result.append({
            "sentence_bleu_score": sentence_bleu_score,
            "rouge1_f1":rouge1_f1,
            "rouge2_f1":rouge2_f1,
            "rougeL_f1":rougeL_f1,
            "bertscore":bertscore,
            # "meteorscore":meteor_score
        })
    # 统一计算corpus BLEU
    corpus_bleu_score = corpus_bleu(all_references, all_candidates,
                                    weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=SmoothingFunction().method3)
    # 计算平均值
    sentence_bleu_score_avg = sum([item["sentence_bleu_score"] for item in result]) / len(result)
    rouge1_f1_avg = sum([item["rouge1_f1"] for item in result]) / len(result)
    rouge2_f1_avg = sum([item["rouge2_f1"] for item in result]) / len(result)
    rougeL_f1_avg = sum([item["rougeL_f1"] for item in result]) / len(result)
    bertscore_avg = sum([item["bertscore"] for item in result]) / len(result)
    # meteorscore = sum([item["meteorscore"] for item in result]) / len(result)
    result.append({
        "corpus_bleu_score_avg": corpus_bleu_score,
        "sentence_bleu_score_avg": sentence_bleu_score_avg,
        "rouge1_f1_avg":rouge1_f1_avg,
        "rouge2_f1_avg":rouge2_f1_avg,
        "rougeL_f1_avg":rougeL_f1_avg,
        "bertscore_avg":bertscore_avg,
        # "meteorscore":meteorscore
    })
    result_dir = f"{args.output_dir}/{args.model}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(f"{result_dir}/{args.reason}.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Result saved to {result_dir}/{args.reason}.json")

if __name__=="__main__":
    args = eval_parser_args()
    compute_metrics(args)
    # reference = "package 'VehicleDefinition' {\n\tprivate import ScalarValues::*;\n\tpart def Vehicle {\n\t\tattribute mass : Real;\n\t\tattribute status : VehicleStatus;\n\t\tpart eng : Engine;\n\t\tref part driver : Person;\n\t}\n\tattribute def VehicleStatus {\n\t\tattribute gearSetting : Integer;\n\t\tattribute acceleratorPosition : Real;\n\t}\n\tpart def Engine;\t\n\tpart def Person;\n}"
    # candidate = "package VehicleManagement {\n\n  value definition Percentage is Real {\n    unit = \"%%\"\n  }\n\n  enum definition GearSetting {\n    literal PARK;\n    literal REVERSE;\n    literal NEUTRAL;\n    literal DRIVE;\n    literal LOW;\n  }\n\n  part definition Engine {\n    attribute displacement: Real;\n    attribute maxPower: Real;\n  }\n\n  part definition Driver {\n    attribute name: String;\n    attribute licenseId: String;\n  }\n\n  part definition VehicleStatus {\n    attribute gearSetting: GearSetting;\n    attribute acceleratorPedalPosition: Percentage;\n  }\n\n  part definition Vehicle {\n    attribute mass: Real;\n    part status: VehicleStatus;\n    part engine: Engine;\n    reference driver: Driver[0..1];\n  }\n\n  part definition VehicleManagementSystem {\n    part vehicles: Vehicle[*];\n    part engines: Engine[*];\n    part drivers: Driver[*];\n  }\n}"
    # print(get_bleu_score(candidate,reference))
    # print(get_rough_score(candidate,reference))
    # print(get_menteor_score(candidate,reference))
