"""
Though this story and motivation is compelling, in practice manifold learning techniques tend to be finicky
enough that they are rarely used for anything more than simple qualitative visualization of high-dimensional data.


The following are some of the particular challenges of manifold learning, which all contrast poorly with PCA:

In manifold learning, there is no good framework for handling missing data. In contrast, there are straightforward
iterative approaches for missing data in PCA.

In manifold learning, the presence of noise in the data can "short-circuit" the manifold and drastically change the
embedding. In contrast, PCA naturally filters noise from the most important components.

The manifold embedding result is generally highly dependent on the number of neighbors chosen, and there is generally
no solid quantitative way to choose an optimal number of neighbors. In contrast, PCA does not involve such a choice.

In manifold learning, the globally optimal number of output dimensions is difficult to determine.
In contrast, PCA lets you find the output dimension based on the explained variance.

In manifold learning, the meaning of the embedded dimensions is not always clear. In PCA, the principal
components have a very clear meaning.

In manifold learning the computational expense of manifold methods scales as O[N^2] or O[N^3]. For PCA, there
exist randomized approaches that are generally much faster (though see the megaman package for some more scalable
implementations of manifold learning).


With all that on the table, the only clear advantage of manifold learning methods over PCA is their ability to
preserve nonlinear relationships in the data; for that reason I tend to explore data with manifold methods only
after first exploring them with PCA.
Scikit-Learn implements several common variants of manifold learning beyond Isomap and LLE: the Scikit-Learn
documentation has a nice discussion and comparison of them. Based on my own experience, I would give the following
recommendations:

For toy problems such as the S-curve we saw before, locally linear embedding (LLE) and its variants
(especially modified LLE), perform very well. This is implemented in sklearn.manifold.LocallyLinearEmbedding.

For high-dimensional data from real-world sources, LLE often produces poor results, and isometric mapping
(IsoMap) seems to generally lead to more meaningful embeddings. This is implemented in sklearn.manifold.Isomap

For data that is highly clustered, t-distributed stochastic neighbor embedding (t-SNE) seems to work very well,
though can be very slow compared to other methods. This is implemented in sklearn.manifold.TSNE.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS, LocallyLinearEmbedding
sns.set()


def make_hello(N=1000, rseed=42):
    # создаем рисунок с текстом "Hello"; сохраняем его в формате PNG
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('images\\hello.png')
    plt.close(fig)

    # открываем этот файл PNG и берем из него случайные точки
    from matplotlib.image import imread
    data = imread('images\\hello.png')[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]


def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)


def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(np.dot(C, C.T))
    return np.dot(X, V[:X.shape[1]])


def make_hello_s_curve(X):
    t = (X[:, 0] - 2) * 0.75 * np.pi
    x = np.sin(t)
    y = X[:, 1]
    z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T


"""Вызываем эту функцию и визуализируем полученные данные"""
X = make_hello(2000)
colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], **colorize)
ax.axis('equal')
fig.savefig('images\\mds1.png')


"""
При взгляде на подобные данные становится ясно, что конкретные значения x и y  - не самая существенная 
характеристика этого набора данных: мы можем пропорционально увеличить/сжать или повернуть данные, а 
надпись HELLO все равно останется четко различимой. Напромер, при использовании матрицы вращения для 
вращения данных значения x и y изменятся, но данные, по существу, останутся теми же.
"""
X2 = rotate(X, 20) + 5
fig, ax = plt.subplots()
ax.scatter(X2[:, 0], X2[:, 1], **colorize)
ax.axis('equal')
fig.savefig('images\\mds2.png')


"""Существенным есть расстояние между точками, для этого используют матрицу расстояний"""
D = pairwise_distances(X)
print(X.shape, D.shape)


"""Можем ее визуализировать"""
fig, ax = plt.subplots()
plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar()
fig.savefig('images\\mds3.png')


"""
Сформировав аналогичным образом матрицу расстояний между подвергшимися 
вращению и сдвигу точками, увидим, что она не поменялась
"""
D2 = pairwise_distances(X2)
print(np.allclose(D, D2))


"""MDS служит для обратного преобразования - из матрицы расстояний в D-мерное координатное представление данных"""
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.axis('equal')
fig.savefig('images\\mds4.png')


"""Спроэцируем в 3хмерное пространство"""
X3 = random_projection(X, 3)
print(X3.shape)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2], **colorize)
ax.view_init(azim=70, elev=50)
fig.savefig('images\\mds5.png')


"""Восстановление при помощи MDS"""
model = MDS(n_components=2, random_state=1)
out3 = model.fit_transform(X3)
fig, ax = plt.subplots()
ax.scatter(out3[:, 0], out3[:, 1], **colorize)
ax.axis('equal')
fig.savefig('images\\mds6.png')


"""Рассмотрим следующее вложение, которое деформируется в форму трехмерной буквы S"""
XS = make_hello_s_curve(X)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2], **colorize)
fig.savefig('images\\mds7.png')


"""
Базовые зависимрсти между точками данных сохранены, но на этот раз данные были преобразованы 
нелинейным образом: они были свернуты в форму буквы S. Есои попытаться использовать для этих 
данных простой алгоритм MDS, он не сумеет "развернуть" это нелинейное вложение и мы потеряем 
из видусущественные зависимости во вложенном многообразии.
"""
model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)
fig, ax = plt.subplots()
ax.scatter(outS[:, 0], outS[:, 1], **colorize)
ax.axis('equal')
fig.savefig('images\\mds8.png')


"""
В то время как MDS пытается сохранить расстояния между всемя парами точек, что не практично для
нелинейных преобразований, метод LLE (локальное линейное валожение/locally linear embedding)
сохраняет расстояние между близлежащими точками, что позволяет относительно хорошо "развернуть"
нелинейное преобразование.
"""
model = LocallyLinearEmbedding(n_neighbors=100, n_components=2, method='modified', eigen_solver='dense')
out = model.fit_transform(XS)
fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15)
fig.savefig('images\\mds9.png')


plt.show()
