import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import seaborn as sns
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
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


faces = fetch_lfw_people(min_faces_per_person=30)
print(faces.data.shape)


"""Быстро визуализируем несколько изображений, чтобы посмотреть, с чем мы имеем дело"""
fig, ax = plt.subplots(4, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='gray')
fig.savefig('images\\isomap_faces1.png')


"""Удобно начать с вычисления PCA и изучения полученой доли объясняемой дисперсии"""
model = PCA(100).fit(faces.data)
fig, ax = plt.subplots()
ax.plot(np.cumsum(model.explained_variance_ratio_))
ax.set_xlabel('n components')
ax.set_ylabel('cumulative variance')
fig.savefig('images\\isomap_faces2.png')


"""
Как видим, для сохранения 90% дисперсии необходимо почти 100 компонент. Это значит, что данные, по своей сути,
имеют черезвычайно высокую размерность и их невозможно описать линейно с помощью всего нескольких компонент.
В подобном случае могут оказаться полезны нелинейные вложения на базе многообразий, такие как LLE и Isomap
"""
model = Isomap(n_components=2)
proj = model.fit_transform(faces.data)
print(proj.shape)


"""
Результат представляет собой двумерную проекцию всех исходных изображений. Чтобы лучше представить, 
что говорит нам эта проекция, опишем функцию, выводящуб миниатюры изображений в местах проекций
"""
fig, ax = plt.subplots(figsize=(10, 10))
plot_components(faces.data, model=Isomap(n_components=2), images=faces.images[:, ::2, ::2])
fig.savefig('images\\isomap_faces3.png')


"""Из результата видно, что первые два признака описывают падение света и ракурс сьемки"""


plt.show()
