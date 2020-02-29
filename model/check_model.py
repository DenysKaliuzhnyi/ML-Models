from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut


iris = load_iris()
X = iris.data
y = iris.target
model = KNeighborsClassifier(n_neighbors=1)


"""плохой наивный метод проверки модели"""
# model.fit(X, y)
# y_model = model.predict(X)
# print(accuracy_score(y, y_model))


"""хороший метод проверки модели - отложенный данные (holdout sets)"""
# X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)
# model.fit(X1, y1)
# y2_model = model.predict(X2)
# print(accuracy_score(y2, y2_model))


"""лучший способ проверки модели - перекрестная проверка (cross-validation)"""
# X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)
# y2_model = model.fit(X1, y1).predict(X2)
# y1_model = model.fit(X2, y2).predict(X1)
# print(accuracy_score(y1, y1_model), accuracy_score(y2, y2_model))


"""больше блоков и специальная функция"""
# score = cross_val_score(model, X, y, cv=5)
# print(score)


"""частичный случай предыдущей - проверка по отдельныйм объектам (leave-one-out cross-validation)"""
score = cross_val_score(model, X, y, cv=LeaveOneOut())
print(X, y)
print(score)
print(score.mean())




