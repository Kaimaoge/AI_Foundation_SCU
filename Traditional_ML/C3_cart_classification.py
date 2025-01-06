# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:37:39 2025

@author: AA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# 设置全局字体样式
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18
})

# 生成一个非线性（例如月亮形状）分类数据集
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义绘图的函数
def plot_decision_boundary(model, X, y, ax):
    # 设置网格范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 预测每个网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    ax.contourf(xx, yy, Z, alpha=0.75, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=ListedColormap(['#FF0000', '#0000FF']))
    ax.set_title(model.__class__.__name__, pad=15)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

# 初始化模型
perceptron = Perceptron(max_iter=1000, random_state=42)
rbf_svm = SVC(kernel='rbf', gamma='auto', random_state=42)
cart = DecisionTreeClassifier(random_state=42)

# 拟合模型
perceptron.fit(X_train, y_train)
rbf_svm.fit(X_train, y_train)
cart.fit(X_train, y_train)

# 创建图形
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 绘制每个模型的决策边界
plot_decision_boundary(perceptron, X_train, y_train, axs[0])
plot_decision_boundary(rbf_svm, X_train, y_train, axs[1])
plot_decision_boundary(cart, X_train, y_train, axs[2])

# 显示图形
plt.tight_layout()
plt.savefig('perceptron_svc_cart.pdf')
plt.show()

