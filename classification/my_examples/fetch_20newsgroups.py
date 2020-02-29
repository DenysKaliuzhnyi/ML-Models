import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score


fontsize_titles = 16
fontsize_lables = 24
fontsize_ticks = 16


categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space',  'comp.graphics']
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')
########################################################################################################################
########################################################################################################################
########################################################################################################################
# """custom TfidfVectorizer"""
# tfid = TfidfVectorizer()
# tfid_fit = tfid.fit_transform(train.data)
#
# mltb = MultinomialNB()
# mltb.fit(tfid_fit, train.target)
#
# tfid_new_voc = TfidfVectorizer(vocabulary=tfid.vocabulary_)
# tfid_test = tfid_new_voc.fit_transform(test.data)
#
# y_pred = mltb.predict(tfid_test)
#
# mat = confusion_matrix(test.target, y_pred)
# fig, ax = plt.subplots(figsize=(20, 20))
# sns.heatmap(mat.T,
#             square=True,
#             annot=True,
#             fmt='d',
#             cbar=False,
#             xticklabels=test.target_names,
#             yticklabels=test.target_names,
#             ax=ax)
# plt.title(accuracy_score(test.target, y_pred))
# plt.xlabel('true label')
# plt.xlabel('predicted label')
# plt.tight_layout()
# plt.savefig('images\\polynomial_TfidfVectorizer_custom.png')
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
"""pipeline TfidfVectorizer"""
model = Pipeline(steps=[('TfidfVectorizer', TfidfVectorizer()), ('MultinomialNB', MultinomialNB())])
parameters = {
    'TfidfVectorizer__binary': (True, False),
    'MultinomialNB__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
}
grid = GridSearchCV(model, parameters, cv=7)
grid.fit(train.data, train.target)

new_model = grid.best_estimator_
new_model.fit(train.data, train.target)
y_pred = new_model.predict(test.data)

mat = confusion_matrix(test.target, y_pred)
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(mat.T,
            square=True,
            annot=True,
            fmt='d',
            cbar=False,
            xticklabels=train.target_names,
            yticklabels=train.target_names,
            ax=ax)
plt.title(f"params = {grid.best_params_}, cv={grid.best_score_}, ac={accuracy_score(test.target, y_pred)}",
          fontsize=fontsize_titles)
plt.xlabel('true label', fontsize=fontsize_lables)
plt.ylabel('predicted label', fontsize=fontsize_lables)
ax.tick_params(axis='both', labelsize=fontsize_ticks)
plt.tight_layout()
plt.savefig('images\\polynomial_TfidfVectorizer_pipeline.png')


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# print(predict_category('Sendinwefesfseg a payload to the ISS'))
#
# print(predict_category('discussing islam vs atheism'))
#
# print(predict_category('determining the screen resolution'))
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# """custom CountVectorizer"""
# tfid = CountVectorizer()
# tfid_fit = tfid.fit_transform(train.data)
#
# mltb = MultinomialNB()
# mltb.fit(tfid_fit, train.target)
#
# tfid_new_voc = CountVectorizer(vocabulary=tfid.vocabulary_)
# tfid_test = tfid_new_voc.fit_transform(test.data)
#
# y_pred = mltb.predict(tfid_test)
#
# mat = confusion_matrix(test.target, y_pred)
# fig, ax = plt.subplots(figsize=(20, 20))
# sns.heatmap(mat.T,
#             square=True,
#             annot=True,
#             fmt='d',
#             cbar=False,
#             xticklabels=test.target_names,
#             yticklabels=test.target_names,
#             ax=ax)
# plt.title(accuracy_score(test.target, y_pred))
# plt.xlabel('true label')
# plt.xlabel('predicted label')
# plt.tight_layout()
# plt.savefig('images\\polynomial_CountVectorizer_custom.png')
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""pipeline CountVectorizer"""
# print(grid.get_params().keys())
model = Pipeline(steps=[('CountVectorizer', CountVectorizer()), ('MultinomialNB', MultinomialNB())])
parameters = {
    'CountVectorizer__binary': (True, False),
    'MultinomialNB__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
}
grid = GridSearchCV(model, parameters, cv=7)
grid.fit(train.data, train.target)

new_model = grid.best_estimator_
new_model.fit(train.data, train.target)
y_pred = new_model.predict(test.data)

mat = confusion_matrix(test.target, y_pred)
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(mat.T,
            square=True,
            annot=True,
            fmt='d',
            cbar=False,
            xticklabels=train.target_names,
            yticklabels=train.target_names,
            ax=ax)
plt.title(f"params = {grid.best_params_}, cv={grid.best_score_}, ac={accuracy_score(test.target, y_pred)}",
          fontsize=fontsize_titles)
plt.xlabel('true label', fontsize=fontsize_lables)
plt.ylabel('predicted label', fontsize=fontsize_lables)
ax.tick_params(axis='both', labelsize=fontsize_ticks)
plt.tight_layout()
plt.savefig('images\\polynomial_CountVectorizer_pipeline.png.png')


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# print(predict_category('Sendinwefesfseg a payload to the ISS'))
#
# print(predict_category('discussing islam vs atheism'))
#
# print(predict_category('determining the screen resolution'))
