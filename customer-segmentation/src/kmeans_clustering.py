from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def kmeans_clustering(scaled_data, n_clusters=3):
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    
    # Silhouette Score
    silhouette_kmeans = silhouette_score(scaled_data, kmeans_labels)
    print(f"Silhouette Score for K-Means: {silhouette_kmeans}")
    
    # Plot Clusters
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans_labels, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.title("K-Means Clustering")
    plt.xlabel("Annual Income (Standardized)")
    plt.ylabel("Spending Score (Standardized)")
    plt.show()
