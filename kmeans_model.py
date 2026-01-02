from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans(data, k=3):
    """
    Standardizes data and applies K-Means clustering.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)

    return labels, kmeans
