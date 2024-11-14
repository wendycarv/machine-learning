import matplotlib.pyplot as plt
import numpy as np

accuracy = np.array([0.67, 0.7382, 0.82, 0.86, 0.8758, 0.909400])
train_points = np.array([100, 200, 500, 750, 1000, 2000])
plt.plot(train_points, accuracy, color = 'r')
plt.title('Number of training points vs accuracy')
plt.xlabel('Number of training points')
plt.ylabel('Accuracy')
plt.show()