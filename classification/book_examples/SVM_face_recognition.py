import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import stats
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs, make_circles
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix
sns.set()


faces = fetch_lfw_people(min_faces_per_person=60)
print(pd.value_counts(faces.target_names[faces.target]))
print(faces.images.shape)


"""Выведем на рисунок несколько из этих лиц, чтобы увидеть, с чем мы будеи иметь дело"""
fig, ax = plt.subplots(3, 5, figsize=(9, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
fig.tight_layout()
fig.savefig('images\\SVM_face_recognition1.png')


"""
Рассмотрим каждый пиксел как признак, однако их достаточно много, 
поэтому выделим самые важные (150 из 3000) с помощью PCA
"""
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced', gamma='auto')
model = make_pipeline(pca, svc)

# cv_result = cross_val_score(model,
#                             faces.data,
#                             faces.target,
#                             cv=30,
#                             n_jobs=-1)
# print(cv_result)

Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

param_grid = dict(svc__C=[1, 5, 10, 50],
                  svc__gamma=[0.0001, 0.0005, 0.001, 0.005])
grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(Xtrain, ytrain)
print(grid.best_params_)


"""Делаем предсказания на тестовых данных"""
model = grid.best_estimator_
yfit = model.predict(Xtest)


"""Рассмотрим гекоторые из контрольных изображений и предсказаных для них значений"""
fig, ax = plt.subplots(4, 6, figsize=(10, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted names; Incorrect Labels in Red', size=14)
fig.savefig('images\\SVM_face_recognition2.png')


"""
Чтобы лучше прочувствовать эффективность работы нашего оценивателя, воспользуемся отчетом 
о классификации, в котором приведена статистика восстановления значений по каждой метке
"""
"""
precision
Precision is the ability of a classiifer not to label an instance positive that is actually negative.
For each class it is defined as as the ratio of true positives to the sum of true and false positives.
Said another way, “for all instances classified positive, what percent was correct?”
recall
Recall is the ability of a classifier to find all positive instances. For each class it is defined 
as the ratio of true positives to the sum of true positives and false negatives. Said another way, 
“for all instances that were actually positive, what percent was classified correctly?”
f1 score
The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 
and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy measures as they embed 
precision and recall into their computation. As a rule of thumb, the weighted average of F1 
should be used to compare classifier models, not global accuracy.
F1 Score = 2*(Recall * Precision) / (Recall + Precision) -  What percent of positive predictions were correct? 
support
Support is the number of actual occurrences of the class in the specified dataset. Imbalanced 
support in the training data may indicate structural weaknesses in the reported scores of the 
classifier and could indicate the need for stratified sampling or rebalancing. Support doesn’t 
change between models but instead diagnoses the evaluation process.
The support is the number of samples of the true response that lie in that class.
"""
print(classification_report(ytest, yfit, target_names=faces.target_names))


"""Можем также вывести матрицу различий между этими классами"""
fig, ax = plt.subplots(figsize=(8, 8))
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T,
            square=True,
            annot=True,
            fmt='d',
            cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names,
            ax=ax)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.tight_layout()
fig.savefig('images\\SVM_face_recognition3.png')


plt.show()