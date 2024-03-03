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
        item["context"] = context[item["relevant"]]
        
        if "answer" in item.keys():
            item["answers"] = {"text":[item["answer"]["text"]], "answer_start":[item["answer"]["start"]]}
            item.pop("answer")
        
        item.pop("relevant")
        
        if "paragraphs" in item.keys():
            item.pop("paragraphs")
        
    with open(f"qa_{os.path.basename(args.filename)}", "w") as new_file:
        json.dump(file, new_file, indent=4, ensure_ascii=False)    
    f.close()
    c.close()
    
    f = open(f"qa_{os.path.basename(args.filename)}")
    file = json.load(f)
    print(type(file))
    
if __name__ == "__main__":
    main()