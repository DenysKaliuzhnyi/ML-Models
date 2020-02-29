import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import mode
import seaborn as sns
sns.set()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or None

    # Transform covariance to main axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # draw ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w*w_factor, ax=ax)


X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=0)
X = X[:, ::-1]


gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.savefig("images\\gmm1.png")


"""GMM give a probabilistic estimate"""
probs = gmm.predict_proba(X)
print(probs[:5].round(3))


"""lets visualize probability by size of point"""
size = 50 * probs.max(1) ** 2
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels, s=size, cmap='viridis', edgecolor='w')
plt.savefig("images\\gmm2.png")


"""visualize shape of cluster"""
gmm = GaussianMixture(n_components=4,random_state=42)
fig, ax = plt.subplots()
plot_gmm(gmm, X, ax=ax)
plt.savefig("images\\gmm3.png")


"""lets checkout it's work with stretched data"""
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
rng = np.random.RandomState(13)
X_streched = np.dot(X, rng.randn(2, 2))
fig, ax = plt.subplots()
plot_gmm(gmm, X_streched, ax=ax)
plt.savefig("images\\gmm4.png")


"""first of all the purpose pf GMM is to estimate density of distribution"""
Xmoon , ymoon = make_moons(200, noise=0.05, random_state=0)
fig, ax = plt.subplots()
ax.scatter(Xmoon[:, 0], Xmoon[:, 1])
plt.savefig("images\\gmm5.png")


"""We can see that next application doesn't make sense"""
gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
fig, ax = plt.subplots()
plot_gmm(gmm2, Xmoon, ax=ax)
plt.savefig("images\\gmm6.png")


"""But we can use more clusters in order to estimate a shape of distribution"""
gmm16 = GaussianMixture(n_components=16, covariance_type='full', random_state=0)
fig, ax = plt.subplots()
plot_gmm(gmm16, Xmoon, label=False, ax=ax)
plt.savefig("images\\gmm7.png")


"""And now we can generate new samples based on density"""
Xnew, _ = gmm16.sample(400)
print(Xnew)
fig, ax = plt.subplots()
ax.scatter(Xnew[:, 0], Xnew[:, 1])
plt.savefig("images\\gmm8.png")


"""You can use AIC and BIC to get the optimal number of components (minimum on the graph)"""
n_components = np.arange(1, 21)
models = [GaussianMixture(n_components=n, covariance_type='full', random_state=0).fit(Xmoon) for n in n_components]

fig, ax = plt.subplots()
ax.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')
ax.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.savefig("images\\gmm9.png")


plt.show()