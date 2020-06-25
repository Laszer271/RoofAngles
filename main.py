import pandas as pd
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_excel('Prich jobs stats_old.xlsx', sheet_name='all')
df.drop(columns=['drawn', 'project', 'notes', 'Unnamed: 12'], inplace=True)
corr = df.corr()

df = pd.get_dummies(df)
y1 = df['front roof angle']
#y2 = df['ridgeboard height']
mask = y1.isna() #| y2.isna()
df = df.drop(columns=['front roof angle'])
df = df[~mask]
y1 = y1[~mask]
#y2 = y1[~mask]
df = df.fillna(df.mean())

X = df.values
y = y1.values
X, y = shuffle(X, y)

params = {'n_estimators': 5000, 'max_depth': 20, 'min_samples_split': 10,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
mse = mean_squared_error(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
print("MSE: %.4f" % mse)
print("MAE: %.4f" % mae)