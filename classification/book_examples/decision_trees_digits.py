import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
sns.set()


"""Скачиваем данные о цифрах"""
digits = load_digits()
print(digits.keys())


"""Визуализируем часть данных"""
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))
fig.savefig('images\\decision_tree_digits1.png')


"""Быстро классифицировать цифры с помощью случайного леса можно следующим образом"""
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)


"""Взглянем на отчет для классификации для данного классфикатора"""
print(classification_report(ypred, ytest))


"""В дополнение нарисуем матрицу различий"""
fig, ax = plt.subplots()
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, ax=ax)
plt.xlabel('true label')
plt.ylabel('predicted label')
fig.savefig('images\\decision_tree_digits2.png')


plt.show()