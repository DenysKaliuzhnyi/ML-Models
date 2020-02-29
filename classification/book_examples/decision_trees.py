"""
This section contained a brief introduction to the concept of ensemble estimators, and in particular the
random forest – an ensemble of randomized decision trees. Random forests are a powerful method with several advantages:

Both training and prediction are very fast, because of the simplicity of the underlying decision trees.
In addition, both tasks can be straightforwardly parallelized, because the individual trees are entirely
independent entities.

The multiple trees allow for a probabilistic classification: a majority vote among estimators gives an estimate
of the probability (accessed in Scikit-Learn with the predict_proba() method).

The nonparametric model is extremely flexible, and can thus perform well on tasks that are under-fit by other estimators


A primary disadvantage of random forests is that the results are not easily interpretable: that is, if you would
like to draw conclusions about the meaning of the classification model, random forests may not be the best choice.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
sns.set()


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()

    # рисуем обучающие точки
    ax.scatter(X[:, 0],
               X[:, 1],
               c=y,
               s=30,
               cmap=cmap,
               edgecolor='white',
               clim=(y.min(), y.max()),
               zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # обучаем оцениватель
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # создаем цветной граффик с результатами
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx,
                           yy,
                           Z,
                           alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap,
                           clim=(y.min(), y.max()),
                           zorder=1)
    ax.set(xlim=xlim, ylim=ylim)


X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow', edgecolor='white')
fig.savefig('images\\decision_tree1.png')


tree = DecisionTreeClassifier().fit(X, y)

fig, ax = plt.subplots()
visualize_classifier(DecisionTreeClassifier(), X, y, ax=ax)
fig.savefig('images\\decision_tree2.png')


plt.show()



