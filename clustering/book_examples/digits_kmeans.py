import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from scipy.stats import mode
import seaborn as sns
sns.set()


digits = load_digits()
print(digits.data.shape)


kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)


"""centers of clusters are also 64-D point, so we can interpret them as 'typical' cluster's digit"""
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, centers in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(centers, interpolation='nearest', cmap=plt.cm.binary)
fig.savefig("images\\digits_kmeans1")


"""checkout real clusters and calculate score"""
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

print(accuracy_score(digits.target, labels))


fig, ax = plt.subplots()
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T,
            square=True,
            annot=True,
            fmt='d',
            cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names,
            ax=ax)
ax.set_xlabel('true label')
ax.set_ylabel('predicted label')
fig.savefig("images\\digits_kmeans2")


"""preprocess data with t-SNE"""
tsne = TSNE(n_components=2, init='pca', random_state=0)
digits_proj = tsne.fit_transform(digits.data)

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

print(accuracy_score(digits.target, labels))


plt.show()
