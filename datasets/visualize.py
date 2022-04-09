import matplotlib.pyplot as plt
import numpy as np

dataset = input('Choose a dataset: ')
prefix = './' + dataset + '/'

data = np.load(prefix + 'statistics.npy')

plt.subplot(1, 3, 1)
plt.hist(data[0], bins=1000, color='red')
plt.xlabel('Time Length')

plt.subplot(1, 3, 2)
plt.hist(data[1][np.argwhere(data[1] < 150)], bins=100, color='blue')
plt.xlabel('Number of Interactions (less than 150)')

plt.subplot(1, 3, 3)
plt.plot(data[0], data[1], '.', color='green')
plt.xlabel('Time')
plt.ylabel('Interactions')

plt.show()

