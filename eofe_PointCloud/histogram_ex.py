import matplotlib.pyplot as plt
import numpy as np

num = 10
X = np.arange(num)
W = np.random.randint(1, num*2, num)

hist = plt.hist(X, bins=num, weights=W, density=False, cumulative=False, label='A',
                range=(X.min()-1, X.max()+1), color='r', edgecolor='black', linewidth=1.2)

plt.title('scatter', pad=10)
plt.xlabel('X axis', labelpad=10)
plt.ylabel('Y axis', labelpad=20)

plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in', pad=8, top=True, right=True)

plt.show()