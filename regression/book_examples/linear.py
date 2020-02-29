import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process import GaussianProcessRegressor
sns.set()


"""Simple linear regression"""
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)

model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(xfit, yfit)
ax.set(title=f"Model slope = {model.coef_[0]}, model intercept = {model.intercept_}")
fig.savefig("images\\image1.png")

X1 = 10 * rng.rand(100, 3)
y1 = 0.5 + np.dot(X1, [1.5, -2, 1])
model.fit(X1, y1)
print(model.intercept_, model.coef_)


"""Polynomial basis functions"""
x2 = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
features = poly.fit_transform(x2[:, np.newaxis])
print(features)

poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
rng = np.random.RandomState(1)
x3 = 10 * rng.rand(50)
y3 = np.sin(x3) + 0.1 * rng.randn(50)
poly_model.fit(x3[:, np.newaxis], y3)
yfit = poly_model.predict(xfit[:, np.newaxis])
fig, ax = plt.subplots()
ax.scatter(x3, y3)
ax.plot(xfit, yfit)
fig.savefig("images\\image2.png")


"""Gaussian basis functions"""
class GaussianFeatures(BaseEstimator, TransformerMixin):
    """равномерно распределенные Гауссовы признаки для одномерных вхлдных данных"""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gaussian_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # создаем N центров, распределенных по всему диапазону данных
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gaussian_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)


gauss_model = make_pipeline(GaussianFeatures(20), LinearRegression())
gauss_model.fit(x3[:, np.newaxis], y3)
yfit = gauss_model.predict(xfit[:, np.newaxis])
fig, ax = plt.subplots()
ax.scatter(x3, y3)
ax.plot(xfit, yfit)
ax.set(xlim=(0, 10))
fig.savefig("images\\image3.png")


gb = GaussianProcessRegressor()
gb.fit(x3[:, np.newaxis], y3)
yfit = gb.predict(xfit[:, np.newaxis])
fig, ax = plt.subplots()
ax.scatter(x3, y3)
ax.plot(xfit, yfit)
ax.set(xlim=(0, 10))
fig.savefig("images\\image4.png")


def basis_plot(model, title=None, img=0):
    fig, ax = plt.subplots(2, sharex='col')
    model.fit(x3[:, np.newaxis], y3)
    ax[0].scatter(x3, y3)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    if title:
        ax[0].set_title(title)
    ax[1].plot(model.steps[0][1].centers_, model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location', ylabel='coefficient', xlim=(0, 10))
    fig.savefig(f"images\\image{img}.png")


model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model, img=5)


"""Регуляризация или Ridge regression L2"""

"""
For instance many elements used in the objective function of
a learning algorithm (such as the RBF kernel of Support Vector
Machines or the L1 and L2 regularizers of linear models) assume that
all features are centered around 0 and have variance in the same
order. If a feature has a variance that is orders of magnitude larger
that others, it might dominate the objective function and make the
estimator unable to learn from other features correctly as expected.
"""
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
basis_plot(model, title='Ridge regression', img=6)


"""Лассо-регуляризация L1, по возможности делает коефициенты нулями"""
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
basis_plot(model, title='Lasso regression', img=7)

plt.show()