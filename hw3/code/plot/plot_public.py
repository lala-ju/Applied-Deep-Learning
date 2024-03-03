from matplotlib import pyplot as plt

ppl_un = [3.67823, 3.46591, 3.66788, 3.90255]
ppl = [3.70658, 3.50497, 3.71518, 3.952907]
epoch = [1, 2, 3, 4]
        
plt.clf()
plt.xticks(epoch)
plt.plot(epoch, ppl_un, marker='o', label="unprocessed dataset") 
plt.plot(epoch, ppl, marker='o', label="processed dataset") 
plt.xlabel("epochs") 
plt.ylabel("mean ppl score") 
plt.legend()
plt.savefig("ppl_public.png")
plt.clf()


