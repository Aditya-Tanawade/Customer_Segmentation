from data_preprocessing import load_and_preprocess_data
from eda import exploratory_data_analysis
from kmeans_clustering import kmeans_clustering
from dbscan_clustering import dbscan_clustering

# Load and preprocess data
data, scaled_data = load_and_preprocess_data('C:/Users/ARJUN/Desktop/BDA_Project/customer-segmentation/data/synthetic_customer_data.csv')


# Perform Exploratory Data Analysis
exploratory_data_analysis(data)

# Perform K-Means Clustering
kmeans_clustering(scaled_data)

# Perform DBSCAN Clustering
dbscan_clustering(scaled_data)
