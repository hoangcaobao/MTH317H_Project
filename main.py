import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#read file csv
df=pd.read_csv("housing.csv")

#create X and y matrix
X=df.iloc[:, :-2]
X=np.array(X)
y=df.iloc[:, -2:-1]
y=np.array(y)

#add one to each data of X 
one=np.ones((X.shape[0], 1))
X=np.concatenate((X, one), axis=1)

#split data
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

#solution for equation gradient equal 0
w=np.dot(np.dot(np.linalg.pinv(np.dot(X_train.T, X_train)), X_train.T), y_train)

#test data
df = pd.DataFrame({'median_house_value': y_test.flatten(),
                   'predicted': np.dot(X_test, w).flatten()})
df.to_csv("result.csv",index=False)
