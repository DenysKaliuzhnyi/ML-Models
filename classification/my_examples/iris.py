import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


"""Скачиваем и форматируем данные"""
iris = load_iris()
X = iris.data
y = iris.target
fnames = iris.feature_names
tnames = iris.target_names

data = pd.DataFrame(np.hstack([X, y[:, np.newaxis]]),
                    columns=np.hstack([iris.feature_names, ['species']]))
data['species'] = data['species'].replace(dict(zip([0, 1, 2], tnames)))

g = sns.relplot(x=fnames[0], y=fnames[2], s=50, hue='species', hue_order=tnames, data=data)


"""Строим модель"""
pipe = Pipeline([('scale', MinMaxScaler()),
                 ('model', GaussianNB())])
model = pipe
model.fit(X[:, [0, 2]], y)


rng = np.random.RandomState(0)
# Xnew = rng.rand(2000, 2)
# for i in range(4):
# minmax_scale = MinMaxScaler(feature_range=(X[:, 0].min(), X[:, 0].max()))
# Xnew[:, 0] = minmax_scale.fit_transform(Xnew[:, 0][:, np.newaxis]).flatten()
# minmax_scale = MinMaxScaler(feature_range=(X[:, 2].min(), X[:, 2].max()))
# Xnew[:, 1] = minmax_scale.fit_transform(Xnew[:, 1][:, np.newaxis]).flatten()
# ynew = model.predict(Xnew)

datanew = pd.DataFrame(np.hstack([X, y[:, np.newaxis]]),
                       columns=[iris.feature_names[0], iris.feature_names[2], 'species'])
datanew['species'] = datanew['species'].replace(dict(zip([0, 1, 2], tnames)))

g.map(sns.scatterplot, x=fnames[0], y=fnames[2], hue='species', hue_order=tnames, alpha=0.2, data=datanew)
g.set_xlabels(fnames[0])
g.set_ylabels(fnames[2])
plt.tight_layout()

plt.savefig('images\\image1')










