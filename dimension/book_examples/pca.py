"""
In this section we have discussed the use of principal component analysis for dimensionality reduction,
for visualization of high-dimensional data, for noise filtering, and for feature selection within high-dimensional data.
Because of the versatility and interpretability of PCA, it has been shown to be effective in a wide variety of contexts
and disciplines. Given any high-dimensional dataset, I tend to start with PCA in order to visualize the relationship
between points (as we did with the digits), to understand the main variance in the data (as we did with the eigenfaces),
and to understand the intrinsic dimensionality (by plotting the explained variance ratio). Certainly PCA is not useful
for every high-dimensional dataset, but it offers a straightforward and efficient path to gaining insight into
high-dimensional data.
PCA's main weakness is that it tends to be highly affected by outliers in the data. For this reason, many robust
variants of PCA have been developed, many of which act to iteratively discard data points that are poorly described
by the initial components. Scikit-Learn contains a couple interesting variants on PCA, including RandomizedPCA
and SparsePCA, both also in the sklearn.decomposition submodule. RandomizedPCA, which we saw earlier, uses a
non-deterministic method to quickly approximate the first few principal components in very high-dimensional data,
while SparsePCA introduces a regularization term (see In Depth: Linear Regression) that serves to enforce sparsity
of the components.
In the following sections, we will look at other unsupervised learning methods that build on some of the ideas of PCA.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
sns.set()


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0,
                      shrinkB=0,
                      color='black')
    ax.annotate('', v0, v1, arrowprops=arrowprops)


"""Легче всего визуализировать его поведение на примере двумерного набора данных"""
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1])
ax.axis('equal')
fig.savefig("images\\pca1.png")


pca = PCA(n_components=2)
pca.fit(X)
print("components =", pca.components_)
print("explained variance =", pca.explained_variance_)


"""
Чтобы понять смысл этих чисел, визуализируем их в виде векторов над входными данными, используя 
компоненты для задания направления векторов, а объяснимую дисперсию - в качестве квадратов их длин
"""
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_ + v, pca.mean_)
    draw_vector(pca.mean_ - v, pca.mean_)
ax.axis('equal')
fig.savefig("images\\pca2.png")


"""Пример исползования PCA в качестве понижающего размерность преобразования"""
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:", X.shape)
print("transformed shape:", X_pca.shape)


"""
Для лучшего понимания эффекта этого понижения размерности можно выполнить 
обратное преобразование этих данных и нарисовать их рядом с исходными
"""
fig, ax = plt.subplots()
X_new = pca.inverse_transform(X_pca)
ax.scatter(X[:, 0], X[:, 1], alpha=0.2)
ax.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
ax.axis('equal')
fig.savefig("images\\pca3.png")


plt.show()
