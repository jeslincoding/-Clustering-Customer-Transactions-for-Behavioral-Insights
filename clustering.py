import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from preprocessing import load_and_preprocess_data

def perform_clustering(data):
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    kmeans_silhouette = silhouette_score(data, kmeans_labels)
    
    # Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=3)
    agglomerative_labels = agglomerative.fit_predict(data)
    agglo_silhouette = silhouette_score(data, agglomerative_labels)
    
    return {
        "kmeans_labels": kmeans_labels,
        "kmeans_silhouette": kmeans_silhouette,
        "agglomerative_labels": agglomerative_labels,
        "agglo_silhouette": agglo_silhouette,
    }

if __name__ == "__main__":
    data = load_and_preprocess_data("data/dataset.csv")
    results = perform_clustering(data)
    print(f"K-Means Silhouette Score: {results['kmeans_silhouette']}")
    print(f"Agglomerative Silhouette Score: {results['agglo_silhouette']}")
