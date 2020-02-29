"""
Because naive Bayesian classifiers make such stringent assumptions about data,
they will generally not perform as well as a more complicated model. That said,
they have several advantages:

They are extremely fast for both training and prediction

They provide straightforward probabilistic prediction

They are often very easily interpretable

They have very few (if any) tunable parameters

These advantages mean a naive Bayesian classifier is often a good choice as an initial
baseline classification. If it performs suitably, then congratulations: you have a very fast,
very interpretable classifier for your problem. If it does not perform well, then you can begin
exploring more sophisticated models, with some baseline knowledge of how well they should perform.


Naive Bayes classifiers tend to perform especially well in one of the following situations:

When the naive assumptions actually match the data (very rare in practice)

For very well-separated categories, when model complexity is less important

For very high-dimensional data, when model complexity is less important

The last two points seem distinct, but they actually are related: as the dimension
of a dataset grows, it is much less likely for any two points to be found close together
(after all, they must be close in every single dimension to be close overall). This means
that clusters in high dimensions tend to be more separated, on average, than clusters in low
dimensions, assuming the new dimensions actually add information. For this reason, simplistic
classifiers like naive Bayes tend to work as well or better than more complicated classifiers
as the dimensionality grows: once you have enough data, even a simple model can be very powerful.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB


"""
наивные баесовские модели - група исключительно быстрых и простых алгоритмов классификации, 
зачастую подходящих для наборов данных очень высоких размерностей. В силу их быстроты и столь небольшого
числа настраиваемых параметров они оказываются оченб удобны в качестве грубого эталона для задач классификации.
Наивность предпологает независимость всех признаков 
"""

"""
Вероятно, самый простой для понимания наивный юайесовский классификатор - Гауссов. 
В этом классификаторе  допущение состоит в том, что данные всех категорний взяты из 
простого нормального распределения (без ковариации между измерениями)
"""
fig, ax = plt.subplots()
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.savefig('images\\gaussian1')

model = GaussianNB()
model.fit(X, y)

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
ax.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.savefig('images\\gaussian2')
"""
Положительная сторона этого байесовского формального представления 
заключается в возможности естественной вероятности классификатора
"""
yprob = model.predict_proba(Xnew)
print(yprob.round(2))







plt.show()