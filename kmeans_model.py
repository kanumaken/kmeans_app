"""
kmeans_model.py

This module contains the logic for preprocessing data
and applying the K-Means clustering algorithm.
"""

# Import K-Means algorithm from scikit-learn
from sklearn.cluster import KMeans

# Import scaler to normalize feature values
from sklearn.preprocessing import StandardScaler


def run_kmeans(data, k=3):
    """
    Standardizes the dataset and applies K-Means clustering.

    Parameters:
    - data (DataFrame): Input dataset with numerical features
    - k (int): Number of clusters to form (default is 3)

    Returns:
    - labels: Cluster labels for each data point
    - kmeans: Trained K-Means model
    """

    # Create a scaler to normalize the data
    scaler = StandardScaler()

    # Scale the data so all features contribute equally
    scaled_data = scaler.fit_transform(data)

    # Initialize K-Means with a fixed random state for reproducibility
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit the model and assign cluster labels
    labels = kmeans.fit_predict(scaled_data)

    # Return the labels and the trained model
    return labels, kmeans
