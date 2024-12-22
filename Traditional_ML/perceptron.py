# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:31:51 2024

@author: AA
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def perceptron_update_gif_with_final_boundary(num_points=20, steps=100, filename="perceptron_update_with_final.gif"):
    """
    Generate a GIF of the perceptron update process with linearly separable data.

    Parameters:
    - num_points: Number of random points to generate
    - steps: Number of updates to simulate
    - filename: Output GIF filename
    """
    # Generate linearly separable data
    np.random.seed(42)
    X_pos = np.random.uniform(3, 5, (num_points // 2, 2))  # Points for class +1
    X_neg = np.random.uniform(-5, 3, (num_points // 2, 2))  # Points for class -1
    X = np.vstack((X_pos, X_neg))
    y = np.array([1] * (num_points // 2) + [-1] * (num_points // 2))

    # Initialize weights and bias
    w = np.random.uniform(-1, 1, 2)
    b = 0

    frames = []

    for step in range(steps):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot points
        for i in range(num_points):
            color = 'green' if y[i] == 1 else 'red'
            marker = 'o' if np.sign(np.dot(w, X[i]) + b) == y[i] else 'x'
            ax.scatter(X[i, 0], X[i, 1], color=color, marker=marker, s=100, zorder=5)

        # Draw decision boundary
        x = np.linspace(-6, 6, 100)
        if w[1] != 0:
            y_line = -(w[0] * x + b) / w[1]
        else:
            y_line = np.zeros_like(x)
        ax.plot(x, y_line, 'b--', label="Decision Boundary")

        # Find a misclassified point
        misclassified = [(X[i], y[i]) for i in range(num_points) if np.sign(np.dot(w, X[i]) + b) != y[i]]
        if misclassified:
            x_i, y_i = misclassified[0]
            # Update weights and bias
            w += y_i * x_i
            b += y_i

        # Add arrows for weight vector
        ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='blue', label=r'$\mathbf{w}$')

        # Set plot limits and labels
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Step {step + 1}: Perceptron Update", fontsize=16)

        # Save frame
        frame_filename = f"frame_{step}.png"
        plt.tight_layout()
        plt.savefig(frame_filename)
        frames.append(frame_filename)
        plt.close()

        # Stop if all points are correctly classified
        if not misclassified:
            break

    # Add final frame to show the converged decision boundary
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(num_points):
        color = 'green' if y[i] == 1 else 'red'
        ax.scatter(X[i, 0], X[i, 1], color=color, marker='o', s=100, zorder=5)
    if w[1] != 0:
        y_line = -(w[0] * x + b) / w[1]
    else:
        y_line = np.zeros_like(x)
    ax.plot(x, y_line, 'r-', label="Final Decision Boundary")
    ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='blue', label=r'$\mathbf{w}$')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Final Decision Boundary", fontsize=16)
    final_frame_filename = f"final_frame.png"
    plt.tight_layout()
    plt.savefig(final_frame_filename)
    frames.append(final_frame_filename)
    plt.close()

    # Create GIF
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(filename, images, duration=0.5)

    # Clean up temporary frames
    for frame in frames:
        os.remove(frame)

    print(f"GIF saved as {filename}")

# Run the perceptron update GIF generation with linearly separable data
perceptron_update_gif_with_final_boundary(num_points=20, steps=100, filename="perceptron_update_with_final.gif")
