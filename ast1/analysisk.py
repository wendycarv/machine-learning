import matplotlib.pyplot as plt
import numpy as np

accuracy = np.array([0.8458, 0.734600, 0.8311, 0.7886, 0.7995, 0.778, 0.7337, 0.64, 0.5946, 0.5284])
k_amt = np.array([1, 2, 3, 4, 5, 10, 20, 50, 70, 100])
plt.plot(k_amt, accuracy, color = 'b')
plt.title('Value of k vs accuracy')
plt.xlabel('Value of k')
plt.ylabel('Accuracy')
plt.show()