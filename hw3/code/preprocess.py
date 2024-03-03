import json
import argparse
from utils import get_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

with open(args.data, 'r') as file:
    stuff = file.read()
    datas = json.loads(stuff)


for data in datas:
    data['instruction'] = get_prompt(data["instruction"])
    if 'output' in data.keys():
        data['output'] = data["output"].split("答案：")[-1] if "答案：" in data["output"] else data["output"]

with open(args.output, 'w') as file:
    json.dump(datas, file, indent=4, ensure_ascii=False)