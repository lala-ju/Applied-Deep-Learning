from matplotlib import pyplot as plt

# I record the loss in the txt file, need to parse the txt file to get the loss
epoch = [1, 2, 3, 4, 5]
lines = []
with open("all_loss.txt", "r") as file:
    lines = file.readlines()

loss = []
for text in lines:
    loss.append(float(text.split(": ")[-1]))

plt.xticks(epoch)
plt.plot(epoch, loss, marker='o') 
plt.xlabel("epoch") 
plt.ylabel("loss on train set") 
plt.savefig("train_loss.png")

lines = []
with open("all_match.txt", "r") as file:
    lines = file.readlines()

match = []
for text in lines:
    metric = text.split("{")[-1]
    exact_match = metric.split(",")[0]
    match.append(float(exact_match.split(": ")[-1]))

plt.clf()
plt.xticks(epoch)
plt.plot(epoch, match, marker='o') 
plt.xlabel("epoch") 
plt.ylabel("exact match on valid set") 
plt.savefig("valid_match.png")