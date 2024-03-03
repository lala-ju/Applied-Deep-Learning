import json
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

loss = []
eval_loss = []
steps = []
eval_steps = []

with open(args.data, 'r') as file:
    stuff = file.read()
    datas = json.loads(stuff)
    
history = datas['log_history']
for record in history:
    if "loss" in record.keys():
        loss.append(record['loss'])
        steps.append(record['step'])
    elif "eval_loss" in record.keys():
        eval_loss.append(record['eval_loss'])
        eval_steps.append(record['step'])
        
plt.xticks(steps)
plt.plot(steps, loss, marker='o') 
plt.xlabel("steps") 
plt.ylabel("training loss") 
plt.savefig("loss_train.png")
plt.clf()

plt.xticks(eval_steps)
plt.plot(eval_steps, eval_loss, marker='o') 
plt.xlabel("steps") 
plt.ylabel("eval loss") 
plt.savefig("evalloss_train.png")

