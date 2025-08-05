import os

import json

with open("dataset/sysml/dataset.json","r",encoding="utf-8") as f:
    data = json.load(f)

grammar_list = []
for item in data:
    if item["grammar"] not in grammar_list:
        grammar_list.append(item["grammar"])

print(grammar_list)

