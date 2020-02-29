import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_sample_image
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from scipy.stats import mode
import seaborn as sns

sns.set()


def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data

    # choose random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)


china = load_sample_image('china.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china)
plt.tight_layout()
plt.savefig('images\\compress_image_kmeans1.png')


"""format is (height, width, RGB)"""
print(china.shape)


"""consider data as color-points in 3-D and normalize in to measure from 0 to 1"""
data = china / 255.0
data = data.reshape(-1, 3)
print(data.shape)


"""visualize 10000 of this points"""
plot_pixels(data, title='Input color space: 16 million possible colors')
plt.savefig('images\\compress_image_kmeans2.png')


"""lets reduce 16 million colors to 16 by MiniBatchKmeans as we have lots of samples"""
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors, title='Reduced color space: 16 colors')
plt.savefig('images\\compress_image_kmeans3.png')


"""lets compare two color schemes"""
china_recolored = new_colors.reshape(china.shape)
fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16)
plt.savefig('images\\compress_image_kmeans4.png')


plt.show()