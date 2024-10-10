from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def dbscan_clustering(scaled_data, eps=0.3, min_samples=5):
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(scaled_data)
    
    # Silhouette Score
    silhouette_dbscan = silhouette_score(scaled_data, dbscan_labels)
    print(f"Silhouette Score for DBSCAN: {silhouette_dbscan}")
    
    # Plot Clusters
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=dbscan_labels, cmap='plasma')
    plt.title("DBSCAN Clustering")
    plt.xlabel("Annual Income (Standardized)")
    plt.ylabel("Spending Score (Standardized)")
    plt.show()
