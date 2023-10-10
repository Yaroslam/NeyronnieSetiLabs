import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
class KMeans:
    def __init__(self, n_clusters=8, max_iterations=300):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for i in range(self.max_iterations):
            clusters = [[] for _ in range(self.n_clusters)]
            for x in X:
                closest_centroid = self._closest_centroid(x)
                clusters[closest_centroid].append(x)

            prev_centroids = self.centroids
            self.centroids = self._calculate_centroids(clusters)

            if self._has_converged(prev_centroids, self.centroids):
                return clusters

    def predict(self, X):
        labels = []
        for x in X:
            closest_centroid = self._closest_centroid(x)
            labels.append(closest_centroid)
        return labels

    def _init_centroids(self, X):
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        for i in range(self.n_clusters):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def _closest_centroid(self, x):
        distances = [np.linalg.norm(x - c) for c in self.centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _calculate_centroids(self, clusters):
        n_features = len(clusters[0][0])
        centroids = np.zeros((self.n_clusters, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(cluster, axis=0)
            centroids[i] = centroid
        return centroids

    def _has_converged(self, prev_centroids, centroids):
        distances = [np.linalg.norm(prev_centroids[i] - centroids[i]) for i in range(self.n_clusters)]
        return np.sum(distances) == 0

df = pd.read_csv('iris.csv')
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit(df.to_numpy())


iris_setosa = df.loc[df["variety"] == 1]
iris_virginica = df.loc[df["variety"] == 2]
iris_versicolor = df.loc[df["variety"] == 3]

sns.FacetGrid(df,
              hue="variety",
              ).map(sns.distplot,
                          "petal.length").add_legend()
sns.FacetGrid(df,
              hue="variety",
              ).map(sns.distplot,
                          "petal.width").add_legend()
sns.FacetGrid(df,
              hue="variety",
             ).map(sns.distplot,
                          "sepal.length").add_legend()
plt.show()

silhouette_avg = silhouette_score(df.to_numpy(), clusters)
print("Средний коэффициент силуэта:", silhouette_avg)


# setosa = 1
# "Versicolor" = 2
# "Virginica"  = 3




