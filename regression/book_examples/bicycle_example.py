import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, Normalizer, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.metrics import r2_score, accuracy_score
import seaborn as sns
sns.set()


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.1f' % x)


def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Рассчитываем длительность светового дня для заданой даты"""
    days = (date - pd.datetime(2000, 12, 21)).days
    m = 1. - np.tan(np.radians(latitude)) * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.24))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.


counts = pd.read_csv('datasets\\Fremont_Bridge_Bicycle_Counter.csv', index_col='Date', parse_dates=True)

"""вычтслим ежедневный поток велосипедов"""
daily = counts.resample('d').sum()
daily = daily[['Fremont Bridge Total']]
daily.columns = ['Total']

"""добавим двоичный столбцы-индикаторы дня недели"""
for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
    daily[day] = (daily.index.dayofweek == i).astype(float)

"""добавим индикаторы праздничных дней"""
cal = USFederalHolidayCalendar()
daily = daily.join(pd.Series(1,
                             index=cal.holidays('2010', '2020'),
                             name='holiday'))
daily['holiday'].fillna(0, inplace=True)

"""добавим длительность севетового дня"""
daily['daylight_hrs'] = hours_of_daylight(daily.index)
# daily[['daylight_hrs']].plot()

# """дбавим признаки по четвертям года"""
# quarters = pd.get_dummies(daily.index.quarter)
# quarters.index = daily.index
# quarters.columns = ['quarter1', 'quarter2', 'quarter3', 'quarter4']
# daily = daily.merge(quarters, left_index=True, right_index=True)


"""Скачиваем данные о погоде"""
weather = pd.read_csv('datasets\\1990777.csv',
                      index_col='DATE',
                      usecols=['DATE', 'TMIN', 'TMAX', 'PRCP'],
                      parse_dates=True)

"""добавим среднюю температуру и индикатор засушливого дня"""
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])
weather['dry_day'] = (weather['PRCP'] == 0).astype(int)

daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry_day']])

"""добавим счетчик по годам"""
daily['annual'] = ((daily.index - daily.index[0]) / 365.).days


"""строим модель линейной регрессии"""
colnames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday',
            'daylight_hrs', 'PRCP', 'dry_day', 'Temp (C)', 'annual']
X = daily[colnames]
y = daily['Total']


pipe = Pipeline([('scale', StandardScaler()),
                 ('features', PolynomialFeatures()),
                 ('model', Ridge())])

param_grid = dict(scale=[StandardScaler(), MinMaxScaler()],
                  features__degree=(1, 2, 3),
                  features__include_bias=[True, False],
                  features__interaction_only=[True, False],
                  model__fit_intercept=[True, False],
                  model__alpha=np.arange(4, 10))
grid = GridSearchCV(pipe, param_grid=param_grid, cv=7)

grid.fit(X, y)
print(grid.best_params_, grid.best_score_)
model = grid.best_estimator_


daily['predicted'] = model.predict(X)
params = pd.Series(model.named_steps['model'].coef_,
                   index=model.named_steps['features'].get_feature_names(colnames)).round()

err = np.std([model.fit(*resample(X, y)).named_steps['model'].coef_ for i in range(100)], 0)

print(pd.DataFrame({'effect': params.round(0),
                    'error': err.round(0)}))

print("r2_score =",
      r2_score(y_true=daily['Total'], y_pred=daily['predicted']),
      '=',
      model.score(X, y))

print(daily.head())

"""сделаем к-блочную перекресную проверку"""
kf = KFold(n_splits=20, shuffle=True, random_state=1)
cv_result = cross_val_score(model,
                            X,
                            y,
                            cv=kf,
                            n_jobs=-1)
print("к-блочная перекресная проверка:", cv_result.mean())


"""сравним с фикстивным регрессором (среднее)"""
dummy = make_pipeline(MinMaxScaler(), DummyRegressor(strategy='mean'))
dummy.fit(X, y)
cv_result_dummy = cross_val_score(dummy,
                                  X,
                                  y,
                                  cv=kf,
                                  n_jobs=-1)
print("к-блочная перекресная проверка для фиктивного регрессора:", cv_result_dummy.mean())


sns.relplot('Total', 'predicted', data=daily)
plt.savefig('images\\image1_1')
daily[['Total', 'predicted']].plot(alpha=0.5)
plt.savefig('images\\image1_2')


plt.show()
