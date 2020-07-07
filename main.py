import random

def loadData(fileName):
    data = []
    outputs = []
    with open(fileName) as file:
        reader = file.readlines()
        for row in reader:
            l = row.split(",")
            lable = l[-1][0:-1]
            l = l[:4]
            data.append(l)
            outputs.append(lable)
    return data, outputs


def nrTeste(matrice, outputs):
    matriceTeste = []
    outputsTeste = []
    nrT = 0.2 * len(matrice)
    for _ in range(int(nrT)):
        ra = random.randrange(len(matrice) - 1)
        matriceTeste.append(matrice[ra])
        outputsTeste.append(outputs[ra])
        matrice.pop(ra)
        outputs.pop(ra)
    return matriceTeste, outputsTeste


inputs,outputs=loadData("iris.data")
inputsTest,outputsTest=nrTeste(inputs,outputs)
print("inputs---------------------------",inputs)
print("outputs---------------------------------",outputs)
labelNames = list(set(outputs))



from sklearn.cluster import KMeans

unsupervisedClassifier = KMeans(n_clusters=3, random_state=0)
unsupervisedClassifier.fit(inputs)
computedTestIndexes = unsupervisedClassifier.predict(inputsTest)
computedTestOutputs = [labelNames[value] for value in computedTestIndexes]

from sklearn.metrics import accuracy_score
print("acc: ", accuracy_score(outputsTest, computedTestOutputs))