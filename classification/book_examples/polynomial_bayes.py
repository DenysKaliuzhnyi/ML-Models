import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix


"""
наивные баесовские модели - група исключительно быстрых и простых алгоритмов классификации, 
зачастую подходящих для наборов данных очень высоких размерностей. В силу их быстроты и столь небольшого
числа настраиваемых параметров они оказываются оченб удобны в качестве грубого эталона для задач классификации.
Наивность предпологает независимость всех признаков 
"""

"""
имеет допущение о том, что признаки сгенерированы на основе простого плиномиального распределения. 
Полиномальный наивный байесовский классификатор нередко используется при классификации текста, 
где признаки соответствуют количеству слов или частотам их употребления в классифицируемых документах" 
"""
categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space',  'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
print(train.data[5])

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(train.data, train.target)
labels = model.predict(test.data)

mat = confusion_matrix(test.target, labels)
fig, ax = plt.subplots()
sns.heatmap(mat.T,
            square=True,
            annot=True,
            fmt='d',
            cbar=False,
            xticklabels=train.target_names,
            yticklabels=train.target_names,
            ax=ax)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.tight_layout()
plt.savefig('images\\palynomial1')


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


print(predict_category('Sending a payload to the ISS'))

print(predict_category('discussing islam vs atheism'))

print(predict_category('determining the screen resolution'))


plt.show()