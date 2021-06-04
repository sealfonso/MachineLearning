import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from scipy.optimize import curve_fit
'exec(%matplotlib inline)'

# # LINEAR
# x = np.arange(-5.0, 5.0, 0.1)
# y = 2*(x) + 3
# y_noise = 2*np.random.normal(size=x.size)
# ydata = y + y_noise
# fig1 = plt.figure(1)
# plt.plot(x, ydata, 'bo')
# plt.plot(x, y, 'r')
# plt.ylabel('Dependent Variable')
# plt.xlabel('Independet Variable')

# # CUBIC
# y1 = 1*(x**3) + 1*(x**2) + 1*x + 3
# y1_noise = 20 * np.random.normal(size=x.size)
# ydata1 = y1 + y1_noise
# fig2 = plt.figure(2)
# plt.plot(x, ydata1,  'bo')
# plt.plot(x,y1, 'r') 
# plt.ylabel('Dependent Variable')
# plt.xlabel('Indepdendent Variable')

# # QUADRATIC
# y2 = np.power(x,2)
# y2_noise = 2 * np.random.normal(size=x.size)
# ydata2 = y2 + y2_noise
# fig3 = plt.figure(3)
# plt.plot(x, ydata2,  'bo')
# plt.plot(x,y2, 'r') 
# plt.ylabel('Dependent Variable')
# plt.xlabel('Indepdendent Variable')

# # EXPONENTIAL
# y3= np.exp(x)
# fig4 = plt.figure(4)
# plt.plot(x,y3) 
# plt.ylabel('Dependent Variable')
# plt.xlabel('Indepdendent Variable')

# # LOGARITHMIC
# y4 = np.log(x)
# fig5 = plt.figure(5)
# plt.plot(x,y4) 
# plt.ylabel('Dependent Variable')
# plt.xlabel('Indepdendent Variable')

# # SIGMOIDAL/LOGISTIC
# y5 = 1-4/(1+np.power(3, x-2))
# fig6 = plt.figure(6)
# plt.plot(x,y5) 
# plt.ylabel('Dependent Variable')
# plt.xlabel('Indepdendent Variable')
# plt.show()

df = pd.read_csv("china_gdp.csv")
df.head(10)

x_data, y_data = (df["Year"].values, df["Value"].values)
plt.figure(figsize=(8,5))
plt.plot(x_data, y_data, 'ro')
plt.xlabel("Year")
plt.ylabel("Value")


def sigmoid(x, Beta_1, Beta_2):
	y = 1/(1 + np.exp(-Beta_1*(x - Beta_2)))
	return y

xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

popt, pcov = curve_fit(sigmoid, xdata, ydata)
print("beta_1 = %f, beta_2 = %f" % (popt[0],popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


# FOR ACCURACY OF THE MODEL:
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt1, pcov1 = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt1)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , test_y))

