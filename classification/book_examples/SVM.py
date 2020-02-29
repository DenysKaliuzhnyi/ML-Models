"""
We have seen here a brief intuitive introduction to the principals behind support vector
machines. These methods are a powerful classification method for a number of reasons:

Their dependence on relatively few support vectors means that they
are very compact models, and take up very little memory.

Once the model is trained, the prediction phase is very fast.

Because they are affected only by points near the margin, they work well with
high-dimensional data—even data with more dimensions than samples, which is a
challenging regime for other algorithms.

Their integration with kernel methods makes them very versatile, able to adapt
to many types of data.


However, SVMs have several disadvantages as well:

The scaling with the number of samples N is O[N^3] at worst, or O[N^2] for
efficient implementations. For large numbers of training samples, this
computational cost can be prohibitive.

The results are strongly dependent on a suitable choice for the softening parameter C.
This must be carefully chosen via cross-validation, which can be expensive as datasets grow in size.

The results do not have a direct probabilistic interpretation. This can be estimated
via an internal cross-validation (see the probability parameter of SVC), but this extra
estimation is costly.


With those traits in mind, I generally only turn to SVMs once other simpler, faster, and
less tuning-intensive methods have been shown to be insufficient for my needs. Nevertheless,
if you have the CPU cycles to commit to training and cross-validating an SVM on your data,
the method can lead to excellent results.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import stats
import seaborn as sns
sns.set()
from sklearn.datasets.samples_generator import make_blobs, make_circles
from sklearn.svm import SVC


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """строи график решающей функции для двумерной SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # создаем координитную сетку для оценки модели
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # рисуем границы принятия решений и отступы
    ax.contour(X,
               Y,
               P,
               colors='k',
               levels=[-1, 0, 1],
               alpha=0.5,
               linestyles=['--', '-', '--'])

    # рисуем опорные векторы
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300,
                   linewidths=1,
                   edgecolors='k',
                   facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_3D(X, y, ax, elev=30, azim=30):
    r = np.exp(-(X ** 2).sum(1))
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


"""
Здесь мы рассматриваем порождающую классификацию. Вместо моделирования каждого из 
классов мы найдем прямую или кривую (многообразие), отделяющее классы друг от друга
"""
fig, ax = plt.subplots()
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
fig.savefig('images\\SVM1.png')


"""
Линейный разделяющий классификатор попытается провести прямую линию, разделяющуу
два набора данных, создав таким образом модель для классификации. Однако сразу 
же возникает проблема: существует более одной идеально разделяющей два класса прямой.
"""
fig, ax = plt.subplots()
xfit = np.linspace(-1, 3.5)
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
ax.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    ax.plot(xfit, m * xfit + b, '-k')
ax.set(xlim=(-1, 3.5))
fig.savefig('images\\SVM2.png')


"""
Метод опорных векторов предоставляет решение этой проблемы. Идея заключается в следуюзем:
вместо того чтобы рисовать между классами прямую нулевой ширины, можно нарисовать около 
каждой из прямых отступ (margin) некоторой ширины, простирающийся до ближайшей точки.
В методе опорных веторов в качестве оптимальной модели выбирается линия, максимизирующая
этот отступ. Метод опорных веторов - пример оценивания с максимальныи отступом (maximum margin estimator)
"""
fig, ax = plt.subplots()
xfit = np.linspace(-1, 3.5)
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    ax.plot(xfit, yfit, '-k')
    ax.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
ax.set(xlim=(-1, 3.5))
fig.savefig('images\\SVM3.png')


"""Обучим эти же данные на рельном SVM классификаторе"""
fig, ax = plt.subplots()
model = SVC(kernel='linear', C=1E10)
m = model.fit(X, y)
print(m)

ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model, ax=ax)
fig.savefig('images\\SVM4.png')


"""Рассмотрим данные которые не допускают линейного разделения"""
fig, ax = plt.subplots()
X, y = make_circles(100, factor=.1, noise=.1, random_state=1)
clf = SVC(kernel='linear').fit(X, y)
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False, ax=ax)
fig.savefig('images\\SVM5.png')


"""
Попытаемся спроецировать эти данные в пространство более высокой размерности, 
поэтому линейного разделителя будет достаточно. Например, одна из подходящих 
простых проекций -вычисление радиальной базисной функции, центрированной
ро середине совокупности данных
"""
ax = plt.subplot(projection='3d')
plot_3D(X, y, ax=ax)
fig.savefig('images\\SVM6.png')


"""
Для того чтобы автоматически находить лучшие базисные функции используем 
процедуру преобразования ядра, которая задается путем kernel='rbf'
"""
fig, ax = plt.subplots()
clf = SVC(kernel='rbf', C=1E6, gamma='auto')
clf.fit(X, y)
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, ax=ax)
# ax.scatter(clf.support_vectors_[:, 0],
#            clf.support_vectors_[:, 1],
#            s=300,
#            lw=1,
#            facecolors='none')
fig.savefig('images\\SVM7.png')


"""Рассмотрим случай пересекающихся данных"""
fig, ax = plt.subplots()
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
fig.savefig('images\\SVM8.png')


"""
На этот случай есть небольшой поправочный параметр для размытия отступа. 
Данный параметр разрешает некоторым точкам заходить на отступ в тех случаях,
когда это приводит к лучшей аппроксимации. При очень большом значении параметра
С отступ является "жестким" и точки не могут находиться на нем. При меньшем его 
значении отступ становиться более размытым и может включать а себя некоторые точки
"""
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, ax=axi)
    axi.set_title('C = {0:.1f}'.format(C), size=14)
fig.savefig('images\\SVM9.png')


plt.show()
