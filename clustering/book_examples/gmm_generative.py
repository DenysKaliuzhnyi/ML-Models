import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import mode
import seaborn as sns
sns.set()


def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


"""lets try to generate more samples of digits using GMM"""
digits = load_digits()
print(digits.data.shape)
plot_digits(digits.data)
plt.savefig('images\\gmm_generative1.png')


"""GMM may has problems with convergence in high-dimensional space. So, lets apply PCA"""
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)
print(data.shape)


"""Now lets find optimal count of components"""
n_components = np.arange(50, 210, 10)
models = [GaussianMixture(n_components=n, covariance_type='full', random_state=0) for n in n_components]
aics = [model.fit(data).aic(data) for model in models]

fig, ax = plt.subplots()
ax.plot(n_components, aics)
plt.savefig('images\\gmm_generative2.png')


"""we will use 110 components"""
gmm = GaussianMixture(110, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)


"""now we can generate 100 new 41-D points"""
data_new, _ = gmm.sample(100)
print(data_new.shape)
digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)
plt.savefig('images\\gmm_generative3.png')


plt.show()
