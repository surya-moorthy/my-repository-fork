import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import math
data = pd.read_csv("Salary_Data.csv")

experience = data["YearsExperience"]
salary = data["Salary"]
#machine learning handle arrays not data frmaes
x = np.array(experience).reshape(-1,1)
y = np.array(salary).reshape(-1,1)
#we use linear regression and fit() for training    

model = LinearRegression()
model.fit(x,y)
regression_mearured_mse = mean_absolute_error(x,y)
print("MSE:",math.sqrt(regression_mearured_mse))
print("R value:",model.score(x,y))
#this is b0
print(model.coef_[0])
#this is b1
print(model.intercept_[0])
plt.scatter(x,y,color="green")
plt.plot(x,model.predict(x),color="black")
plt.title("linear Regression")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()
print("prediction model:",model.predict([[5]]))
