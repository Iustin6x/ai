from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import sklearn
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_squared_error


def loadData(fileName,inputVariabNames, outputVariabName):
    data = pd.read_csv(fileName)
    inputs = data[inputVariabNames].values.tolist()
    outputs = data[outputVariabName].tolist()
    return inputs, outputs


def least_square_multiple(trainInputs,trainOutputs,testInputs,testOutputs):

    regressor = linear_model.LinearRegression()
    # # training the model by using the training inputs and known training outputs
    regressor.fit(trainInputs, trainOutputs)

    b0, b1, b2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print("using libraries:")
    print('the learnt model: f(x) = ', b0, ' + ', b1, ' * x1 + ', b2, ' *x2')

    test_lst_multiple(regressor, testInputs, testOutputs)


def test_lst_multiple(regressor,testInputs,testOutputs):
    # makes predictions for test data
    computedTestOutputs = regressor.predict(testInputs)
   # plotData([], [], testInputs, computedTestOutputs, testInputs, testOutputs, "predictions vs real test data")

    # compute the differences between the predictions and real outputs

    error = mean_squared_error(testOutputs, computedTestOutputs)
    print("prediction error (tool): ", error)
    print("prediction error manual: ")
    print("l1")
    print(errorL1(testOutputs,computedTestOutputs,len(testOutputs)))
    print("l2")
    print(errorL2(testOutputs, computedTestOutputs, len(testOutputs)))

#Mean absolute error
def errorL1(realOutputs, computedOutputs, noOfSamples):
    return sum(abs(ro - co) for ro, co in zip(realOutputs, computedOutputs)) / noOfSamples

#root mean squared error
def errorL2(realOutputs, computedOutputs, noOfSamples):
    return sqrt(sum((ro - co) ** 2 for ro, co in zip(realOutputs, computedOutputs)) / noOfSamples)


inputs,outputs=loadData('https://raw.githubusercontent.com/lauradiosan/AI-2019-2020/master/lab07/data/world-happiness-report-2017.csv',['Economy..GDP.per.Capita.','Freedom'],'Happiness.Score')
trainInputs,testInputs,trainOutputs,testOutputs=sklearn.model_selection.train_test_split(inputs,outputs, test_size=0.20, random_state=2)

least_square_multiple(trainInputs,trainOutputs,testInputs,testOutputs)