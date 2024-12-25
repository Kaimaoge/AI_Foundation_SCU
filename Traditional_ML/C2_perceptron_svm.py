# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 14:24:44 2024

@author: AA
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_blobs

# Create linearly separable data
X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42, cluster_std=1.0)
y = 2 * y - 1  # Convert labels to -1 and 1

# Train SVM
svm = SVC(kernel="linear", C=1.0)
svm.fit(X, y)

# Train Perceptron
perceptron = Perceptron()
perceptron.fit(X, y)

# Plot settings for consistent style without LaTeX
plt.figure(figsize=(8, 5))
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12
})

# Scatter plot for the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50, label='Data Points')

# Plot the SVM decision boundary
w_svm = svm.coef_[0]
b_svm = svm.intercept_[0]
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals_svm = -(w_svm[0] * x_vals + b_svm) / w_svm[1]
plt.plot(x_vals, y_vals_svm, 'b-', label='SVM Boundary')

# Plot the Perceptron decision boundary
w_perceptron = perceptron.coef_[0]
b_perceptron = perceptron.intercept_[0]
y_vals_perceptron = -(w_perceptron[0] * x_vals + b_perceptron) / w_perceptron[1]
plt.plot(x_vals, y_vals_perceptron, 'r--', label='Perceptron Boundary')

# Labeling
plt.title('Decision Boundaries of SVM and Perceptron')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('perceptron_svm.pdf')
plt.show()
