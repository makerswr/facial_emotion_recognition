import numpy as np
import matplotlib.pyplot as plt
a = []
b = []
with open('./AnalyzedData/HappyAnalyzedData.txt') as f:
    for line in f:
        x,y = line.split()
        a.append(int(x))
        b.append(int(y))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Plot title")
ax1.set_xlabel('x label')
ax1.set_ylabel('y label')
ax1.scatter(a,b)
plt.gca().invert_yaxis()

leg = ax1.legend()

plt.show()
