# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:40:29 2025

@author: AA
"""

import numpy as np
import matplotlib.pyplot as plt

# Define a simple class for the CART regression tree
class CARTRegressionTree:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Stopping criteria
        if len(X) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return {"value": np.mean(y)}

        # Find the best split
        best_split = self._find_best_split(X, y)
        if best_split["gain"] == 0:
            return {"value": np.mean(y)}

        # Split the data
        left_idx = best_split["indices_left"]
        right_idx = best_split["indices_right"]

        # Recursively build the tree
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_subtree,
            "right": right_subtree,
        }

    def _find_best_split(self, X, y):
        best_split = {"gain": 0}
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = np.where(X[:, feature] <= threshold)[0]
                right_idx = np.where(X[:, feature] > threshold)[0]

                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

                gain = self._calculate_variance_reduction(y, left_idx, right_idx)
                if gain > best_split["gain"]:
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "gain": gain,
                        "indices_left": left_idx,
                        "indices_right": right_idx,
                    }
        return best_split

    def _calculate_variance_reduction(self, y, left_idx, right_idx):
        total_variance = np.var(y) * len(y)
        left_variance = np.var(y[left_idx]) * len(left_idx)
        right_variance = np.var(y[right_idx]) * len(right_idx)
        return total_variance - (left_variance + right_variance)

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X])

    def _predict_row(self, row, tree):
        if "value" in tree:
            return tree["value"]
        if row[tree["feature"]] <= tree["threshold"]:
            return self._predict_row(row, tree["left"])
        else:
            return self._predict_row(row, tree["right"])

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
# Generate synthetic data for regression
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)  # Feature: Random numbers between 0 and 5
y = np.sin(X).ravel() + np.random.normal(scale=0.1, size=X.shape[0])  # Target: Sine wave with noise

# Split the data into training and testing sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train the CART regression tree
cart_tree = CARTRegressionTree(min_samples_split=5, max_depth=3)
cart_tree.fit(X_train, y_train)

# Predict on training and testing data
y_train_pred = cart_tree.predict(X_train)
y_test_pred = cart_tree.predict(X_test)

# Calculate mean squared error
train_mse = np.mean((y_train - y_train_pred) ** 2)
test_mse = np.mean((y_test - y_test_pred) ** 2)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="True Data", color="lightgray", s=30)
plt.plot(X, cart_tree.predict(X), label="CART Prediction", color="blue", linewidth=2)
plt.title("CART Regression Tree")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('CART_regression.pdf')
plt.show()

# Display the MSE for training and testing
train_mse, test_mse
