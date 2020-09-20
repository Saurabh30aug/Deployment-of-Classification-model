import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset=pd.read_csv(r"E:\Deployment3\iris.csv")
X=dataset.iloc[:,[0,2,4,6]]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)



#pickle.dump(log_reg,open('timble3.pkl','wb'))

























