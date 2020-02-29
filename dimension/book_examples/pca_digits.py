import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
sns.set()


def plot_digits(data):
    fig, axes = plt.subplots(4,
                           10,
                           figsize=(10, 4),
                           subplot_kw=dict(xticks=[], yticks=[]),
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary',
                  interpolation='nearest',
                  clim=(0, 16))


digits = load_digits()
print(digits.data.shape)


pca = PCA(2)
projected = pca.fit_transform(digits.data)
print(projected.shape)


"""Можно построить граффик двух главных компонент каждой точки"""
fig, ax = plt.subplots()
plt.scatter(projected[:, 0],
            projected[:, 1],
            c=digits.target,
            edgecolor='none',
            alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('component1')
plt.ylabel('component2')
plt.colorbar(ax=ax)
plt.savefig('images\\pca_digits1.png')


"""Выбор количества компонент"""
pca = PCA().fit(digits.data)
fig, ax = plt.subplots()
ax.plot(np.cumsum(pca.explained_variance_ratio_))
ax.set_xlabel('number of components')
ax.set_ylabel('cumulative explained variance')
fig.savefig('images\\pca_digits2.png')


"""Фильтрация шума (добавим его)"""
plot_digits(digits.data)
plt.savefig('images\\pca_digits3.png')


np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)
plt.savefig('images\\pca_digits4.png')


"""
Визуально очевидно, что изображения зашумлены и содержат фиктивные пикселы. Обучим алгоритм 
PCA на этих зашумленных данных, указав, что проекция должна сохранять 50% дисперсии
"""
pca = PCA(0.50).fit(noisy)
print("n_components_ = ", pca.n_components_)


components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)
plt.savefig('images\\pca_digits5.png')


plt.show()