from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve, learning_curve, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


def make_data(N, err=1.0, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


X, y = make_data(40)


"""визуализируем данные с несколькими аппроксимациями их многочленами ращзличной степени"""
fig, ax = plt.subplots()
X_test = np.linspace(-0.1, 1.1, 500)[:, None]
plt.scatter(X.ravel(), y, color='black')
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    ax.plot(X_test.ravel(), y_test, label=f'degree={degree}')
ax.set(xlim=(-0.1, 1.0), ylim=(-2, 12))
ax.legend(loc='best')
plt.savefig('images\\image1.png')


"""визуализируем кривую проверки для выбора наилучшей степени"""
fig, ax = plt.subplots()
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(),
                                          X,
                                          y,
                                          'polynomialfeatures__degree',
                                          degree,
                                          cv=7)
ax.plot(degree,
        np.median(train_score, 1),
        color='blue',
        label='training score')
ax.plot(degree,
        np.median(val_score, 1),
        color='red',
        label='validation score')
ax.legend(loc='best')
ax.set(ylim=(0, 1), xlabel='degree', ylabel='score')
plt.savefig('images\\image2.png')


"""исходя из граффика лучше всего аппроксимирует многочлен третей степени, покажем его"""
fig, ax = plt.subplots()
ax.scatter(X.ravel(), y)
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
ax.plot(X_test.ravel(), y_test)
plt.savefig('images\\image3.png')


########################################################################################################################
########################################################################################################################
########################################################################################################################


"""проведем аналогичное исследование но для большего набора данных"""
X2, y2 = make_data(200)
fig, ax = plt.subplots()
ax.scatter(X2.ravel(), y2)
plt.savefig('images\\image4.png')


fig, ax = plt.subplots()
degree = np.arange(0, 21)
train_score2, val_score2 = validation_curve(PolynomialRegression(),
                                            X2,
                                            y2,
                                            'polynomialfeatures__degree',
                                            degree,
                                            cv=7)
ax.plot(degree,
        np.median(train_score2, 1),
        color='blue',
        label='training score')
ax.plot(degree,
        np.median(val_score2, 1),
        color='red',
        label='validation score')
ax.plot(degree,
        np.median(train_score, 1),
        color='blue',
        alpha=0.3,
        linestyle='dashed')
ax.plot(degree,
        np.median(val_score, 1),
        color='red',
        alpha=0.3,
        linestyle='dashed')
ax.legend(loc='lower center')
ax.set(ylim=(0, 1), xlabel='degree', ylabel='score')
plt.savefig('images\\image5.png')


########################################################################################################################
########################################################################################################################
########################################################################################################################


"""кривая обучения - график оценки обучения/проверки с учетом размера обучающей последовательности (learning curve)"""
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree),
                                         X,
                                         y,
                                         cv=7,
                                         train_sizes=np.linspace(0.3, 1, 25))
    ax[i].plot(N,
               np.mean(train_lc, 1),
               color='blue',
               label='training score')
    ax[i].plot(N,
               np.mean(val_lc, 1),
               color='red',
               label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]),
                 N[0],
                 N[-1],
                 color='gray',
                 linestyles='dashed')
    ax[i].set(xlim=(N[0], N[-1]),
              ylim=(0, 1),
              xlabel='training size',
              ylabel='score',
              title=f'degree={degree}')
    ax[i].legend(loc='best')
plt.savefig('images\\image6.png')


########################################################################################################################
########################################################################################################################
########################################################################################################################


"""поиск по сетке - один из способов поиска подходящей модели в случае множества гиперпараметров"""
param_grid = {'polynomialfeatures__degree': np.arange(21),
              'linearregression__fit_intercept': [True, False],
              'linearregression__normalize': [True, False]}
grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(X, y)
print(grid.best_params_)

model = grid.best_estimator_
fig, ax = plt.subplots()
ax.scatter(X.ravel(), y)
y_test = model.fit(X, y).predict(X_test)
ax.plot(X_test.ravel(), y_test)
plt.savefig('images\\image7.png')
