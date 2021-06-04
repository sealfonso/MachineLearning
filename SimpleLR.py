import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import r2_score
'exec(%matplotlib inline)'

df=pd.read_csv("FuelConsumption.csv", encoding='unicode_escape')

# take a look at the dataset - head() devuelve las n líneas (filas) del objeto 
df.head()

# summarize the data - da características como cuántos son, la media, desviación estandar...
df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
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


# Se seleccionan el 80% de las filas de manera aleatoria para training
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Se plantea el modelo de regresión
regr = linear_model.LinearRegression()
# asanyarray function is used when we want to convert input to an array but it pass ndarray subclasses through
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
# fit linear model
regr.fit(train_x, train_y)
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

fig5 = plt.figure(5)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("CO2 emissions")

plt.show()

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y - test_y_hat)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y - test_y_hat)**2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))
