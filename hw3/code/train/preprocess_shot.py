import json
import argparse
import random
from utils import get_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--shot', type=int, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--sample', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

with open(args.sample, 'r') as file:
    records = file.read()
    samples = json.loads(records)

with open(args.data, 'r') as file:
    stuff = file.read()
    datas = json.loads(stuff)

tradition = []
modern = []
newline = '\n'

for sample in samples:
    if '文言' in sample['instruction'] or '古代' in sample['instruction'] or '古文' in sample['instruction']:
        tradition.append(sample)
    else:
        modern.append(sample)

for data in datas:
    data['instruction'] = get_prompt(data['instruction'])
    data['output'] = data['output'].split('答案：')[-1] if '答案：' in data['output'] else data['output']
    
    if args.shot <= 0:
        continue
    
    final = ''
    if '文言' in sample['instruction'] or '古代' in sample['instruction'] or '古文' in sample['instruction']:
        length = len(tradition)
        chosen = random.sample(range(0, length-1), args.shot)
        for choice in chosen:
            final += f'{tradition[choice]["instruction"]}{newline}{tradition[choice]["output"]}{newline}'
    else:
        length = len(modern)
        chosen = random.sample(range(0, length-1), args.shot)
        for choice in chosen:
            final += f'{modern[choice]["instruction"]}{newline}{modern[choice]["output"]}{newline}'
    
    final += '以上是之前的翻譯紀錄，現在'
    final += data['instruction']        
    data['instruction'] = final


with open(args.output, 'w') as file:
    json.dump(datas, file, indent=4, ensure_ascii=False)