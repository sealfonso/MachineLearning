import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import r2_score
'exec(%matplotlib inline)'

df=pd.read_csv("FuelConsumption.csv", encoding='unicode_escape')

df.head()

df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'CO2EMISSIONS']]
cdf.head(9)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]
viz.hist()
fig1=plt.figure(1)

fig2=plt.figure(2)
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Fuel Consumption")
plt.ylabel("Emission")

fig3=plt.figure(3)
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")

fig4=plt.figure(4)
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")

plt.show()

# Se seleccionan el 80% de las filas de manera aleatoria para training
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Se plantea el modelo de regresión
regr = linear_model.LinearRegression()
# asanyarray function is used when we want to convert input to an array but it pass ndarray subclasses through
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
# fit linear model
regr.fit(train_x, train_y)
print('Coefficients Case1: ', regr.coef_)


test_y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares (MSE) Case1: %.2f" % np.mean((test_y - test_y_hat)**2))
print("Variance score Case1: %.2f" % regr.score(test_x, test_y))

# Se plantea el modelo de regresión
regr1 = linear_model.LinearRegression()
# asanyarray function is used when we want to convert input to an array but it pass ndarray subclasses through
train_x1 = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
train_y1 = np.asanyarray(train[['CO2EMISSIONS']])
# fit linear model
regr1.fit(train_x1, train_y1)
print('Coefficients Case2: ', regr1.coef_)


test_y1_hat = regr1.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
test_x1 = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
test_y1 = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares (MSE) Case2: %.2f" % np.mean((test_y1 - test_y1_hat)**2))
print("Variance score Case2: %.2f" % regr1.score(test_x1, test_y1))