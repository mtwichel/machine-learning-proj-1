import pandas as pd
import numpy as np
import sys

def makePrediction(model, row):
    ratios = model['ratios']
    totals = model['totals']

    entry = row.to_dict()
    
    predictionTotals = {}
    for targetClass in ratios.keys():
        fields = ratios[targetClass].keys()
        ans = 1
        for field in fields:
            ans *= ratios[targetClass][field][entry[field]]
        ans *= totals[targetClass]
        predictionTotals[targetClass] = ans
    maximumProb = max(predictionTotals.values())
    return list(predictionTotals.keys())[list(predictionTotals.values()).index(maximumProb)]

def makeModel(data):
    print('Training Model')

    # Get proportions for the different classes
    totals = data.groupby(['Class']).count().max(axis=1)
    totals = totals / totals.sum()

    # object that will contain the ratios in the model
    # Structure is ratios[class][field][value] = ratio of that value's occurance in that field
    ratios = {}

    for targetClass in data['Class'].unique():
        classSubsetData = data[data['Class'] == targetClass]
        ratios[targetClass] = {}
        for col in classSubsetData.columns:
            if col != 'Class':
                colTotal = classSubsetData[col].count()
                uniqueVals = classSubsetData[col].unique()
                ratios[targetClass][col] = {}
                for option in uniqueVals:
                    valColTotal = classSubsetData[classSubsetData[col] == option][col].count()
                    ratios[targetClass][col][option] = valColTotal / colTotal
    return {
        'ratios': ratios,
        'totals': totals,
    }

# check for valid filepath
if len(sys.argv) <= 1:
    print('Error: You must enter a filepath as command argument')
    exit()
else:
    dataChoice = sys.argv[1]

if dataChoice == 'iris':
    print('Importing File: iris.csv')
    rawData = pd.read_csv('iris.csv')
    
    data = rawData.drop(['Class'], axis = 1).round(0).applymap(int).applymap(str)

    data['Class'] = rawData['Class']

elif dataChoice == 'house':
    print('Importing File: house-votes-84.csv')
    rawData = pd.read_csv('house-votes-84.csv')
    data = rawData


# split training data
print('Splitting data into test and train')
msk = np.random.rand(len(data)) < 0.8
trainData = data[msk]
testData = data[~msk]

# make model
model = makeModel(data=trainData)

# begin evalutaion process
print('Begining Model Evalutaion')
correctAnswers = 0

for row in testData.iterrows():
    prediction = makePrediction(model, row=row[1].drop(['Class']))
    actual = row[1]['Class']

    if prediction == actual:
        correctAnswers += 1

totalAnswers = testData.shape[0]
accuracy = correctAnswers / totalAnswers
print(accuracy)