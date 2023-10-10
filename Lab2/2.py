import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


def initialize_centroids(data, k):
    n_samples, n_features = data.shape
    centroids = np.zeros((k, n_features))
    for i in range(k):
        centroid = data[np.random.choice(range(n_samples))]
        centroids[i] = centroid
    return centroids


def assign_to_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    return clusters


def calculate_centroids(data, clusters):
    centroids = np.array([np.mean(data[clusters == i], axis=0) for i in range(np.max(clusters) + 1)])
    return centroids


def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)

    for _ in range(max_iterations):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = calculate_centroids(data, clusters)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


m = 1000
k = 3

df = pd.read_csv('iris.csv')

clusters, centroids = kmeans(df.to_numpy(), k)

scatter = plt.scatter(df.to_numpy()[:, 0], df.to_numpy()[:, 1], c=clusters)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red')

labels = ['Кластер {}'.format(label) for label in ['Versicolor', "setosa", "Virginica"]]
plt.legend(handles=scatter.legend_elements()[0], labels=labels)

plt.xlabel('length')
plt.ylabel('width')
plt.title('Результаты кластеризации')

silhouette_avg = silhouette_score(df.to_numpy(), clusters)
print("Средний коэффициент силуэта:", silhouette_avg)
plt.text(0.05, 0.95, f'Средний коэффициент силуэта: {silhouette_avg:.2f}', transform=plt.gca().transAxes)

plt.show()

