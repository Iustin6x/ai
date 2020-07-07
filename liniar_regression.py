import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import math
from sklearn import metrics
dataset = pd.read_csv('iris.csv')

X = dataset.iloc[:,:4].values

Y = dataset['species'].values
print(Y)

X_antr, X_test, Y_antr, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 85)


for j in range(len(X_antr)):
    for i in range(len(X_antr[j])):
        X_antr[j][i]=1/(1 + np.exp(-X_antr[j][i]))

for j in range(len(X_test)):
    for i in range(len(X_test[j])):
        X_test[j][i]=1/(1 + np.exp(-X_test[j][i]))

logisticregression = LogisticRegression()
logisticregression.fit(X_antr, Y_antr)
Y_pred = logisticregression.predict(X_test)

corecte = 0
gresite = 0
for i in range(len(Y_test)):
    print('Expected=',Y_test[i],'    ','Predicted=',Y_pred[i])
    if(Y_test[i]==Y_test[i]):
        corecte+=1
    else:
        gresite+=1

print('Corecte: ', corecte)
print('False:', gresite)       
print("Accuracy:",metrics.accuracy_score(Y_test, Y_test))