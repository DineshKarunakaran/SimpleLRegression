#Vanakkam வணக்கம்
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
file="/home/dinesh/codeit/c++/ML/FuelConsumptionCo2.csv"
df=pd.read_csv(file)
ndf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(ndf.head())
viz=ndf[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(viz.hist())
(plt.show())
#plotting a scatter plot for engine_Size vs emission 
plt.scatter(ndf.ENGINESIZE,ndf.CO2EMISSIONS,color='red')
plt.xlabel("engine size")
plt.ylabel("emission")
plt.show()
mk=np.random.rand(len(df))< 0.8
train=ndf[mk]
test=ndf[~mk]
#simple regression model
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='brown')
plt.xlabel('Engine_size')
plt.ylabel('Emissions')
plt.show()
#Modelling
from sklearn import linear_model
r=linear_model.LinearRegression()
trainx=np.asanyarray(train[['ENGINESIZE']])
trainy=np.asanyarray(train[['CO2EMISSIONS']])
r.fit(trainx,trainy)
print("coefients:",r.coef_)
print("intercept:",r.intercept_)
#plotting output
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='black')
plt.plot(trainx,r.coef_[0][0]*trainx+r.intercept_[0],'-r')
plt.xlabel("engine size")
plt.ylabel("emission")
plt.show()
#evaluation
from sklearn.metrics import r2_score
testx=np.asanyarray(test[['ENGINESIZE']])
testy=np.asanyarray(test[['CO2EMISSIONS']])
tp=r.predict(testx)
print("r2-score:%2f"%r2_score(testy,tp))
