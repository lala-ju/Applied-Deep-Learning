import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="postprocess multiple choice data")
    parser.add_argument("--filename", 
        type=str,
        default=None,
        help="The name of the data to process before entering the model.",
    )
    
    parser.add_argument("--original_file", 
        type=str,
        default=None,
        help="The name of the original data to be referenced.",
    )
    
    parser.add_argument("--context_file", 
        type=str,
        default=None,
        help="The name of the data to be processed.",
    )
    
    args = parser.parse_args()
    assert args.filename is not None, "need to have file to preprocess"
    assert args.original_file is not None, "need to have file to preprocess"
    assert args.context_file is not None, "need to have file to preprocess"
    return args

def main():
    args = parse_args()
    f = open(args.filename)
    file = json.load(f)
    o = open(args.original_file)
    original = json.load(o)
    c = open(args.context_file)
    context = json.load(c)
    
    for i, item in enumerate(file):
        item["context"] = context[original[i]["paragraphs"][item["label"]]]
        item["question"] = original[i]["question"]
        item.pop("label")
        
    with open(f"qa_{os.path.basename(args.original_file)}", "w") as new_file:
        json.dump(file, new_file, indent=4, ensure_ascii=False)    
    f.close()
    c.close()
    o.close()
    
if __name__ == "__main__":
    main()