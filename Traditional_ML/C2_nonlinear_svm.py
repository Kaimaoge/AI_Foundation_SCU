# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 14:46:49 2024

@author: AA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# 生成非线性可分数据
X, y = make_circles(n_samples=300, factor=0.5, noise=0.1, random_state=42)
y = 2 * y - 1  # 将标签转换为-1和1，适配SVM

# 使用RBF核训练SVM
svm = SVC(kernel='rbf', C=1.0, gamma=0.5)
svm.fit(X, y)

# 可视化决策边界
# Update Matplotlib settings to disable LaTeX and use a similar font
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12
})

# Function to plot decision boundary without LaTeX dependency
def plot_decision_boundary_with_style_no_latex(model, X, y, title):
    # Create a grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    # Compute the decision function for each grid point
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margin
    plt.figure(figsize=(8, 5))
    plt.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.8)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=1, linestyles='solid')  # Decision boundary
    
    # Scatter plot of data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50, label='Data Points')
    
    # Labels and title
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Decision Function Value')
    plt.legend()
    plt.grid(True)
    

plt.tight_layout()
# Call the plot function
plot_decision_boundary_with_style_no_latex(svm, X, y, title="Decision Boundary of SVM with RBF Kernel")
plt.savefig('nonlinear_svm.pdf')
plt.show()
