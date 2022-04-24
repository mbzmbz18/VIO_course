import numpy as np
import matplotlib.pyplot as plt

f = open("../builds/app/data_mu.txt")

x = []
y = []
for line in f:
    data = line.split()
    x.append(float(data[0]))
    y.append(float(data[1]))

plt.plot(x, y, marker='o')
plt.xlim(xmin = -1)
plt.ylim(ymin = -100)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel("iter")
plt.ylabel("lambda")
plt.title('LM damping factor plot')
plt.show()