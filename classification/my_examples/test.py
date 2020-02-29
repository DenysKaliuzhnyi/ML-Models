import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

cv = CountVectorizer(stop_words='english')
tv = TfidfVectorizer(stop_words='english')

a = "cat hat bat splat cat bat hat mat cat"
b = "cat mat cat sat"

a2 = "cat  bat splat pip cat bat mat "
b2 = "cat mat lol pip sat"

cv_score = cv.fit_transform([a, b])
# pickle.dump(cv.vocabulary_, open("feature.pkl", "wb"))


loaded_vec = CountVectorizer(vocabulary=cv.vocabulary_)
print(cv.vocabulary_)


# df1 = pd.DataFrame(cv_score, columns=cv.get_feature_names())
# print(df1)

# tv_score = tv.fit_transform([a, b]).toarray()
# df2 = pd.DataFrame(tv_score, columns=cv.get_feature_names())
# print(tv_score)

cv_score2 = loaded_vec.fit_transform([a2, b2]).toarray()


print(pd.DataFrame(cv_score2, columns=loaded_vec.get_feature_names()))
# mltb = MultinomialNB()
#
# mltb.fit(cv_score, [0, 1])
# print(mltb.predict([[0, 3, 2, 1, 0, 1]]))




import seaborn as sns

data = sns.load_dataset('mpg')

X = data[data.columns[-1]].values.reshape(-1, 1).astype('str')
y = data[data.columns[-2]].values.astype('str')
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# X = vec.fit_transform(X.ravel()).toarray()
# print(X)
# model = MultinomialNB()
score = cross_val_score(model, X.ravel(), y, cv=LeaveOneOut())
print(score)
print(score.mean())



# print(X, y)
# X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)
#
#
# model = Pipeline(steps=[('TfidfVectorizer', TfidfVectorizer()), ('MultinomialNB', MultinomialNB())])
# parameters = {
#     'TfidfVectorizer__binary': (True, False),
#     'MultinomialNB__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
# }
# grid = GridSearchCV(model, parameters, cv=7)
# grid.fit(X1.ravel(), y1)
#
# new_model = grid.best_estimator_
# new_model.fit(X1.ravel(), y1)
# y_pred = new_model.predict(X2.ravel())
#
# mat = confusion_matrix(y2, y_pred)
# fig, ax = plt.subplots()
# sns.heatmap(mat.T,
#             square=True,
#             annot=True,
#             fmt='d',
#             cbar=False,
#             # xticklabels=train.target_names,
#             # yticklabels=train.target_names,
#             ax=ax)
# plt.title(f"params = {grid.best_params_}, cv={grid.best_score_}")
# # plt.xlabel('true label', fontsize=fontsize_lables)
# # plt.ylabel('predicted label', fontsize=fontsize_lables)
# # ax.tick_params(axis='both', labelsize=fontsize_ticks)
# plt.tight_layout()
# plt.savefig('images\\kek.png')


data.dropna(axis=1, inplace=True)
X = data[data.columns[:-2]].values
y = data[data.columns[-2]].values.astype('str')
print(X, y)
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)

model = GaussianNB()

model.fit(X1, y1)

y_pred = model.predict(X2)

print(accuracy_score(y2, y_pred))

