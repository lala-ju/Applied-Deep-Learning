import argparse
import json
import os
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="postprocess multiple qa data")
    parser.add_argument("--filename", 
        type=str,
        default=None,
        help="The name of the data to process before entering the model.",
    )
    
    parser.add_argument("--final_filename", 
        type=str,
        default=None,
        help="The name of the data to be processed.",
    )
    
    args = parser.parse_args()
    assert args.filename is not None, "need to have file to postprocess"
    assert args.final_filename is not None, "need to have file to postprocess"
    return args

def main():
    args = parse_args()
    f = open(args.filename)
    file = json.load(f)
    
    id = ["id"]
    answer = ["answer"]
    for ele in file:
        id.append(ele["id"])
        answer.append(ele["answer"])
    
    row = zip(id, answer)
    
    with open(args.final_filename, "w") as final:
        writer = csv.writer(final)
        for row in row:
            writer.writerow(row)
    
    f.close()
    
if __name__ == "__main__":
    main()