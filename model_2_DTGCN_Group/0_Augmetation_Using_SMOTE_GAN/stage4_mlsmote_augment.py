"""
    The goal of this script is to implement the MLSMOTE algorithm for multi-label oversampling.
    It generates synthetic samples by interpolating between feature vectors and combining label sets.
    The script uses a NearestNeighbors implemented in sklearn.
    The synthetic samples are saved as NumPy arrays for further use.
    The script assumes that the features and labels have been pre-computed and saved in NumPy format.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

# === Stage 2: MLSMOTE for Multi-label Oversampling ===

# Load features and labels from Stage 1
X = np.load("/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/X_features.npy")  # Shape: (N, 512)
print(f"X shape: {X.shape}")
Y = np.load("/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/Y_labels.npy")    # Shape: (N, 5)
print(f"Y shape: {Y.shape}")
# Parameters
n_neighbors = 3        # Number of neighbors to consider for interpolation
n_samples = 100        # Number of synthetic samples to generate

def mlsmote(X, Y, n_neighbors=5, n_samples=100):
    """
    Custom Multi-label SMOTE implementation.
    Generates synthetic samples by interpolating between feature vectors and combining label sets.

    Parameters:
        X: ndarray of shape (N, D), original feature vectors
        Y: ndarray of shape (N, L), multi-hot label vectors
        n_neighbors: int, number of neighbors to consider for each sample
        n_samples: int, number of new synthetic samples to generate

    Returns:
        X_aug: ndarray of shape (N + n_samples, D)
        Y_aug: ndarray of shape (N + n_samples, L)
    """
    X = np.array(X)
    Y = np.array(Y)

    new_X, new_Y = [], []
    nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)

    for _ in range(n_samples):
        idx = np.random.randint(0, len(X))
        x_i, y_i = X[idx], Y[idx]

        neighbors = nn_model.kneighbors([x_i], return_distance=False)[0][1:]  # Exclude self
        nn_idx = np.random.choice(neighbors)
        x_nn, y_nn = X[nn_idx], Y[nn_idx]

        alpha = np.random.rand()
        x_new = x_i + alpha * (x_nn - x_i)
        y_new = np.logical_or(y_i, y_nn).astype(int)

        new_X.append(x_new)
        new_Y.append(y_new)

    return np.vstack([X, new_X]), np.vstack([Y, new_Y])

# Run MLSMOTE and save outputs
X_aug, Y_aug = mlsmote(X, Y, n_neighbors=n_neighbors, n_samples=n_samples)
np.save("/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/X_augmented.npy", X_aug)
np.save("/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/underrepresented_Images/BlueRedWhite/Y_augmented.npy", Y_aug)

print(f"✅ Stage 2 complete: {n_samples} synthetic samples created.")
print(f"New dataset shape: {X_aug.shape}, Labels shape: {Y_aug.shape}")
