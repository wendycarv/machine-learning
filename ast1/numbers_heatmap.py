import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Compute the confusion matrix
conf_matrix = np.array([
    [910, 1, 9, 1, 5, 17, 26, 6, 2, 14],
    [0, 1052, 2, 2, 1, 5, 0, 3, 0, 0],
    [8, 79, 741, 28, 21, 12, 7, 63, 21, 10],
    [5, 17, 8, 883, 4, 52, 6, 10, 27, 18],
    [0, 26, 1, 0, 765, 0, 7, 28, 1, 155],
    [22, 21, 1, 82, 13, 660, 34, 14, 25, 43],
    [16, 24, 9, 0, 28, 15, 863, 1, 1, 0],
    [1, 42, 2, 1, 9, 3, 0, 937, 0, 95],
    [9, 59, 10, 63, 13, 49, 23, 30, 696, 57],
    [6, 8, 2, 14, 60, 6, 5, 54, 1, 805]
])

mask = np.eye(conf_matrix.shape[0], dtype=bool)
# Create a heatmap with seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cbar=False, cmap='Reds', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
