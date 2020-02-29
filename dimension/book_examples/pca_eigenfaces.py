import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
sns.set()


faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)


"""
Выясним, какие главные оси координат охватывают этот набор данных. Поскольку набор данных велик, 
воспользуемся классом RandomizedPCA - содержащийся в нем рандомизированный метод позволяет 
аппроксимировать первые N компонент намного быстрее, чем обычный оцениватель PCA
"""
pca = PCA(150, random_state=0, svd_solver='randomized', whiten=True)
pca.fit(faces.data)


fig, axes = plt.subplots(3,
                         8,
                         figsize=(9, 4),
                         subplot_kw=dict(xticks=[], yticks=[]),
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')
fig.savefig('images\\pca_eigenfaces1.png')


"""Посмотрим на интегральную дисперсию этих компонент, чтобы выяснить, какая доля информациии сохраняется"""
fig, ax = plt.subplots()
ax.plot(np.cumsum(pca.explained_variance_ratio_))
ax.set_xlabel('number of components')
ax.set_ylabel('cumulative explained variance')
fig.savefig('images\\pca_eigenfaces2png')


"""Ради уточнения сравним входные изображения с восстановленными из этих 150 компонент"""
pca = PCA(150, random_state=0, svd_solver='randomized', whiten=True).fit(faces.data)
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)


fig, axes = plt.subplots(2,
                         10,
                         figsize=(10, 2.5),
                         subplot_kw=dict(xticks=[], yticks=[]),
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    axes[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    axes[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
axes[0, 0].set_ylabel('full dim\ninput')
axes[1, 0].set_ylabel('150-dim\nreconstruction')
fig.savefig('images\\pca_eigenfaces3png')


plt.show()
