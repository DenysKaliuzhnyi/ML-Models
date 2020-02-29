import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_blobs, make_moons
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import cdist
import seaborn as sns
sns.set()


def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


def plot_kmeans(kmeans, X, n_clusters=4, rseen=0, ax=None):
    labels = kmeans.fit_predict(X)

    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))


"""don't include y_true"""
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], s=50)
fig.savefig("images\\kmeans1.png")


"""build model"""
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


"""plot kmeans"""
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
fig.savefig("images\\kmeans2.png")


"""plot custom kmeans"""
centers, labels = find_clusters(X, 4)
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
fig.savefig("images\\kmeans3.png")


"""bad result in case of non-linear bounds"""
X, y = make_moons(200, noise=0.05, random_state=0)

labels = KMeans(2, random_state=0).fit_predict(X)
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
fig.savefig("images\\kmeans4.png")


"""use spectral clustering to separate classes"""
model = SpectralClustering(n_clusters=2,
                           affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
fig.savefig("images\\kmeans5.png")


"""kmeans has no probabilistic estimate and requires sphere shape of cluster"""
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=0)
X = X[:, ::-1]


kmeans = KMeans(n_clusters=4, random_state=0)
fig, ax = plt.subplots()
plot_kmeans(kmeans, X, ax=ax)
fig.savefig("images\\kmeans6.png")


"""it badly works with ellipses"""
rng = np.random.RandomState(13)
X_streched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters=4, random_state=0)
fig, ax = plt.subplots()
plot_kmeans(kmeans, X_streched, ax=ax)
fig.savefig("images\\kmeans7.png")


plt.show()