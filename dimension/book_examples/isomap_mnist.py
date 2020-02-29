import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import seaborn as sns
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
sns.set()


def plot_components(data, model, images=None, ax=None, thumb_frac=0.05, cmap='gray'):
    ax = ax or plt.gca()

    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')

    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                #  не отображаем слишком близко разположенные точки
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), proj[i])
            ax.add_artist(imagebox)


mnist = fetch_openml('mnist_784')
print(mnist.data.shape)


"""
Этот набор состоит из 70 000 изображений, каждое размером 784 пиксела, то 
есть 28х28 пикселов. Как и ранее, рассмотримнесколько первых изображений
"""
fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(mnist.data[1250 * i].reshape(28, 28), cmap='gray_r')
fig.savefig('images\\isomap_mnist1.png')


"""
Вычислим с помощью обучения на базе многообразий проекцию для этих данных. Используем 
только 1/30 часть данных: вычисления для полного набора занимают длительное время
"""
data = mnist.data[::30]
target = mnist.target[::30]


model = Isomap(n_components=2)
proj = model.fit_transform(data)
fig, ax = plt.subplots()
plt.scatter(proj[:, 0], proj[:, 1], c=target)
plt.clim(-0.5, 9.5)
plt.savefig('images\\isomap_mnist2.png')


"""
Полученные данные показывают некоторые зависимости, однако точки рассположены слишком 
тесно. Можно получить больше информации, изучая за раз данне лишь об одной цифре.
Выбираем для проекции 1/4 цифр "1"
"""
data = mnist.data[mnist.target == 1][::4]
fig, ax = plt.subplots(figsize=(10, 10))
model = Isomap(n_neighbors=5, n_components=2, eigen_solver='dense')
plot_components(data, model, images=data.reshape((-1, 28, 28)), ax=ax, thumb_frac=0.05, cmap='gray_r')
fig.savefig('images\\isomap_mnist3.png')


plt.show()