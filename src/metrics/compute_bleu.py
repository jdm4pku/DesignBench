from nltk.translate.bleu_score import sentence_bleu,corpus_bleu,SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
def get_bleu_score(candidate,reference):
    reference_tokens = reference.split()
    result_tokens = candidate.split()
    score = corpus_bleu([[reference_tokens]], [result_tokens], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method3)
    return score

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


if __name__=="__main__":
    reference = "package 'VehicleDefinition' {\n\tprivate import ScalarValues::*;\n\tpart def Vehicle {\n\t\tattribute mass : Real;\n\t\tattribute status : VehicleStatus;\n\t\tpart eng : Engine;\n\t\tref part driver : Person;\n\t}\n\tattribute def VehicleStatus {\n\t\tattribute gearSetting : Integer;\n\t\tattribute acceleratorPosition : Real;\n\t}\n\tpart def Engine;\t\n\tpart def Person;\n}"
    candidate = "package VehicleManagement {\n\n  value definition Percentage is Real {\n    unit = \"%%\"\n  }\n\n  enum definition GearSetting {\n    literal PARK;\n    literal REVERSE;\n    literal NEUTRAL;\n    literal DRIVE;\n    literal LOW;\n  }\n\n  part definition Engine {\n    attribute displacement: Real;\n    attribute maxPower: Real;\n  }\n\n  part definition Driver {\n    attribute name: String;\n    attribute licenseId: String;\n  }\n\n  part definition VehicleStatus {\n    attribute gearSetting: GearSetting;\n    attribute acceleratorPedalPosition: Percentage;\n  }\n\n  part definition Vehicle {\n    attribute mass: Real;\n    part status: VehicleStatus;\n    part engine: Engine;\n    reference driver: Driver[0..1];\n  }\n\n  part definition VehicleManagementSystem {\n    part vehicles: Vehicle[*];\n    part engines: Engine[*];\n    part drivers: Driver[*];\n  }\n}"
    print(get_bleu_score(candidate,reference))
    print(get_rough_score(candidate,reference))
    print(get_menteor_score(candidate,reference))
