import csv
import json
from matplotlib import pyplot as plt

rouge1 = []
rouge2 = []
rougeL = []
rougeLsum = []
steps = []

with open("plotdata.csv",  mode ='r') as file:
    # reading the CSV file
    csvFile = csv.DictReader(file)

    # displaying the contents of the CSV file
    for lines in csvFile:
        lines["rouge"] = str(lines["rouge"]).replace("'", '"')
        obj = json.loads(lines["rouge"])
        rouge1.append(obj["rouge1"])
        rouge2.append(obj["rouge2"])
        rougeL.append(obj["rougeL"])
        rougeLsum.append(obj["rougeLsum"])
        steps.append(int(lines["steps"]))
        
plt.xticks(steps)
plt.plot(steps, rouge1, marker='o') 
plt.xlabel("steps") 
plt.ylabel("rouge1 score (f score * 100)") 
plt.savefig("rouge1.png")
plt.clf()

plt.xticks(steps)
plt.plot(steps, rouge2, marker='o') 
plt.xlabel("steps") 
plt.ylabel("rouge2 score (f score * 100)") 
plt.savefig("rouge2.png")
plt.clf()

plt.xticks(steps)
plt.plot(steps, rougeL, marker='o') 
plt.xlabel("steps") 
plt.ylabel("rougeL score (f score * 100)") 
plt.savefig("rougeL.png")
plt.clf()
