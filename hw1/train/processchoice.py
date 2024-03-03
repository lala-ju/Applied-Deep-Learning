import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="preprocess multiple choice data")
    parser.add_argument("--filename", 
        type=str,
        default=None,
        help="The name of the data to process before entering the model.",
    )
    
    parser.add_argument("--context_file", 
        type=str,
        default=None,
        help="The name of the data to be processed.",
    )
    
    args = parser.parse_args()
    assert args.filename is not None, "need to have file to preprocess"
    assert args.context_file is not None, "need to have file to preprocess"
    return args

def main():
    args = parse_args()
    f = open(args.filename)
    file = json.load(f)
    c = open(args.context_file)
    context = json.load(c)
    
    for item in file:
        item["sent1"] = item["question"]
        item["sent2"] = ""
        item["ending0"] = context[item["paragraphs"][0]]
        item["ending1"] = context[item["paragraphs"][1]]
        item["ending2"] = context[item["paragraphs"][2]]
        item["ending3"] = context[item["paragraphs"][3]]
        
        if "relevant" in item.keys():
            item["label"] = item["paragraphs"].index(item["relevant"])
            item.pop("relevant")
            
        item.pop("question")
        item.pop("paragraphs")
        if "answer" in item.keys():
            item.pop("answer")
        
    with open(f"choice_{os.path.basename(args.filename)}", "w") as new_file:
        json.dump(file, new_file, indent=4, ensure_ascii=False)    
    f.close()
    c.close()
    
if __name__ == "__main__":
    main()