import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
'exec(%matplotlib inline)'

df=pd.read_csv("FuelConsumption.csv", encoding='unicode_escape')

# take a look at the dataset - head() devuelve las n líneas (filas) del objeto 
df.head()

# summarize the data - da características como cuántos son, la media, desviación estandar...
df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'CO2EMISSIONS']]
cdf.head(9)

fig1=plt.figure(1)
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# Se seleccionan el 80% de las filas de manera aleatoria para training
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# asanyarray function is used when we want to convert input to an array but it pass ndarray subclasses through
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

clf = linear_model.LinearRegression()
# fit linear model
train_y = clf.fit(train_x_poly,train_y)
print('Coefficients: ', clf.coef_)
print('Intercept: ', clf.intercept_)

fig2=plt.figure(2)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
vectx = np.arange(0.0, 10.0, 0.1)
vecty = clf.intercept_[0] + clf.coef_[0][1]*vectx + clf.coef_[0][2]*np.power(vectx,2) + clf.coef_[0][3]*np.power(vectx,3)
plt.plot(vectx, vecty, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

plt.show()

test_x_poly = poly.fit_transform(test_x)
test_y_hat = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y - test_y_hat)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y - test_y_hat)**2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))

