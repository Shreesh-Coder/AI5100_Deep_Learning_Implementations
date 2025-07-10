import numpy as np
import matplotlib.pyplot as plt

def generate_linearly_separable_dataset(size, separation_level):
    # Class 1 points
    class1_points = np.random.normal(loc=0, scale=1, size=(size // 2, 2))
    class1_labels = np.ones((size // 2, 1))

    # Class 2 points with a clear separation
    class2_points = np.random.normal(loc=separation_level, scale=1, size=(size // 2, 2))
    class2_labels = -1 * np.ones((size // 2, 1))

    # Combine the two classes
    X = np.vstack((class1_points, class2_points))
    y = np.vstack((class1_labels, class2_labels))

    # Shuffle the dataset
    shuffle_indices = np.random.permutation(size)
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    return X, y.flatten()

def initialize_weights(num_features):
    return np.random.rand(num_features)

def plot_dataset_with_both_decision_boundaries(X, y, initial_weights, trained_weights, title):
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', marker='o')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], label='Class 2', marker='x')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot initial decision boundary
    x_initial_boundary = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y_initial_boundary = (-initial_weights[0] / initial_weights[1]) * x_initial_boundary - initial_weights[2] / initial_weights[1]
    plt.plot(x_initial_boundary, y_initial_boundary, '-g', label='Initial Decision Boundary')

    # Plot trained decision boundary
    x_trained_boundary = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y_trained_boundary = (-trained_weights[0] / trained_weights[1]) * x_trained_boundary - trained_weights[2] / trained_weights[1]
    plt.plot(x_trained_boundary, y_trained_boundary, '-r', label='Trained Decision Boundary')

    plt.legend()
    plt.show()

def train_perceptron(X, y, weights, nb_epochs_max):
    for epoch in range(nb_epochs_max):
        nb_changes = 0
        for i in range(X.shape[0]):
            if np.dot(X[i], weights) * y[i] <= 0:
                weights = weights + y[i] * X[i]
                nb_changes += 1
        if nb_changes == 0:
            print('Early stopping at epoch number %d' % epoch)
            break

    print('Number of changes: %d' % nb_changes)
    return weights

# Example usage
separation_levels = [2.0, 4.0, 6.0]

for separation_level in separation_levels:
    X, y = generate_linearly_separable_dataset(1000, separation_level)

    # Initialize weights
    initial_weights = initialize_weights(X.shape[1] + 1)

    # Add bias term to the features
    X_bias = np.hstack((X, np.ones((X.shape[0], 1))))

    # Train perceptron
    trained_weights = train_perceptron(X_bias, y, initial_weights, nb_epochs_max=100)

    # Plot the dataset with both decision boundaries
    plot_dataset_with_both_decision_boundaries(X, y, initial_weights, trained_weights, f'Dataset with Initial and Trained Decision Boundaries (Separation Level: {separation_level})')
