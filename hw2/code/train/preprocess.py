import argparse
import jsonlines
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

objobj = []

with jsonlines.open(args.input) as f:
    for obj in f:
        objobj.append(obj)

with open(args.output, 'w') as f:
    json.dump(objobj, f, indent=4, ensure_ascii=False)