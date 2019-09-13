import pandas as pd
import numpy as np
import sys
import pprint as pp

def makePrediction(model, row):
    ratios = model['ratios']
    totals = model['totals']

    entry = row.to_dict()
    
    predictionTotals = {}
    for targetClass in ratios.keys():
        fields = ratios[targetClass].keys()
        ans = 1
        for field in fields:
            try:
                ans *= ratios[targetClass][field][entry[field]]
            except:
                ans *= 1
        ans *= totals[targetClass]
        predictionTotals[targetClass] = ans
    maximumProb = max(predictionTotals.values())
    return list(predictionTotals.keys())[list(predictionTotals.values()).index(maximumProb)]

def makeModel(data, doShuffling = True):
    print('Training Model')

    if doShuffling:
        numCols = data.shape[1]
        colsToShuffle = numCols / 10
        for i in range(1, int(colsToShuffle)):
            data.iloc[:,i] = np.random.permutation(data.iloc[:,i].values)

    # Get proportions for the different classes
    totals = data.groupby(['Class']).count().max(axis=1)
    totals = totals / totals.sum()

    # pull all the fields used to have a more complete set
    fields = {}
    for targetClass in data['Class'].unique():
        classSubsetData = data[data['Class'] == targetClass]
        fields[targetClass] = []
        for col in classSubsetData.columns:
            if col != 'Class':
                uniqueVals = classSubsetData[col].unique()
                fields[targetClass].extend(uniqueVals)
        fields[targetClass] = list(dict.fromkeys(fields[targetClass]))

    # object that will contain the ratios in the model
    # Structure is ratios[class][field][value] = ratio of that value's occurance in that field
    ratios = {}

    for targetClass in data['Class'].unique():
        classSubsetData = data[data['Class'] == targetClass]
        ratios[targetClass] = {}
        for col in classSubsetData.columns:
            if col != 'Class':
                colTotal = classSubsetData[col].count()
                ratios[targetClass][col] = {}
                for option in fields[targetClass]:
                    valColTotal = classSubsetData[classSubsetData[col] == option][col].count()
                    valColTotal += 1
                    ratios[targetClass][col][option] = valColTotal / colTotal
    return {
        'ratios': ratios,
        'totals': totals,
    }

def evaluateModel(model, testingData):
    # begin evalutaion process
    print('Begining Model Evalutaion')
    targetClasses = testingData['Class'].unique()
    
    metrics = {}

    for targetClass in targetClasses:
        metrics[targetClass] = {
            'falseNegatives': 0,
            'falsePositives': 0,
            'trueNegatives': 0,
            'truePositives': 0,
        }
    correctCount = 0
    for row in testingData.iterrows():
        prediction = makePrediction(model, row=row[1].drop(['Class']))
        actual = row[1]['Class']

        if prediction == actual:
            correctCount += 1
            # true positive
            metrics[prediction]['truePositives'] += 1
            # increment true negative for all others
            for targetClass in targetClasses:
                if targetClass != prediction:
                    metrics[targetClass]['trueNegatives'] += 1
        else:
            # true positive
            metrics[prediction]['falsePositives'] += 1
            # increment true negative for all others
            metrics[actual]['falseNegatives'] += 1

    totalAnswers = testingData.shape[0]


    classMetrics = {}
    for targetClass in targetClasses:
        try:
            classMetrics[targetClass] = {
                'recall': (metrics[targetClass]['truePositives']) / (metrics[targetClass]['truePositives'] + metrics[targetClass]['falsePositives']),
                'precision': (metrics[targetClass]['truePositives']) / (metrics[targetClass]['truePositives'] + metrics[targetClass]['falseNegatives']),
            }
        except ZeroDivisionError:
            if metrics[targetClass]['truePositives'] + metrics[targetClass]['falsePositives'] == 0:
                classMetrics[targetClass] = {
                    'recall': 0,
                    'precision': (metrics[targetClass]['truePositives']) / (metrics[targetClass]['truePositives'] + metrics[targetClass]['falseNegatives']),
                }
            elif metrics[targetClass]['truePositives'] + metrics[targetClass]['falseNegatives'] == 0:
                classMetrics[targetClass] = {
                    'recall': (metrics[targetClass]['truePositives']) / (metrics[targetClass]['truePositives'] + metrics[targetClass]['falsePositives']),
                    'precision': 0,
                }
    return {
        'modelAccuraccy': correctCount / totalAnswers,
        'classMetrics': classMetrics
    }
    


def perform2FoldValidation(data, shuffling):
    # split training data
    msk = np.random.rand(len(data)) < 0.5
    group1 = data[msk]
    group2 = data[~msk]

    # make model
    model1 = makeModel(data=group1, doShuffling=shuffling)

    # make model
    model2 = makeModel(data=group2, doShuffling=shuffling)

    eval1 = evaluateModel(model = model1, testingData=group2)
    eval2 = evaluateModel(model = model2, testingData=group1)

    # combine results from individual models to one collective metrics object
    targetClasses = eval1['classMetrics'].keys()

    combinedClassMetrics = {}
    for targetClass in targetClasses:
        combinedClassMetrics[targetClass] = {
            'recall': (eval1['classMetrics'][targetClass]['recall'] + eval2['classMetrics'][targetClass]['recall']) / 2,
            'precision': (eval1['classMetrics'][targetClass]['precision'] + eval2['classMetrics'][targetClass]['precision']) / 2,
        }
    
    return {
        'modelAccuraccy': (eval1['modelAccuraccy'] + eval2['modelAccuraccy']) / 2,
        'classMetrics': combinedClassMetrics
    }
   

def perform2by5FoldValidation(data, shuffling):
    modelAccuracySum = 0
    models = []
    
    for i in range(0,5):
        models.append(perform2FoldValidation(data=data, shuffling=shuffling))
        
    # combine results from individual models to one collective metrics object
    targetClasses = models[0]['classMetrics'].keys()

    combinedClassMetrics = {}
    for targetClass in targetClasses:
        recallSum = precisionSum = 0
        for model in models:
            recallSum += model['classMetrics'][targetClass]['recall']
            precisionSum += model['classMetrics'][targetClass]['precision']
        combinedClassMetrics[targetClass] = {
            'recall': recallSum / 5,
            'precision': precisionSum / 5,
            'fscore': 2 * (((recallSum / 5) * (precisionSum / 5)) / ((recallSum / 5) + (precisionSum / 5)))
        }
    
    for model in models:
        modelAccuracySum += model['modelAccuraccy']
        
    return {
        'modelAccuraccy': modelAccuracySum / 5,
        'classMetrics': combinedClassMetrics
    }

# 
# BEGINING OF MAIN PROGRAM
# 

# check for valid dataset name
if len(sys.argv) <= 1:
    print('What dataset would you like to use?\n\t1: Iris\n\t2: House Votes 84\n\t3: Breast Cancer\n\t4: Glass\n\t5: Soybean')
    numberInput = input()
    if (numberInput == '1'):
        dataChoice = 'iris'
    elif (numberInput == '2'):
        dataChoice = 'house-votes'
    elif (numberInput == '3'):
        dataChoice = 'breast-cancer'
    elif (numberInput == '4'):
        dataChoice = 'glass'
    elif (numberInput == '5'):
        dataChoice = 'soybean'
else:
    dataChoice = sys.argv[1]

print('Would you like to shuffle 1/10 of the features? (Y/n)')
choice = input()
if choice == 'n' or choice == 'N':
    shuffling = False
else:
    shuffling = True

print('\n')

# switch through dataset names
if dataChoice == 'iris':
    print('Importing File: iris.csv')
    rawData = pd.read_csv('iris.csv')
    
    # clean up iris dataset by applying the floor function to every floating point number
    data = rawData.drop(['Class'], axis = 1).applymap(np.floor).applymap(int).applymap(str)
    data['Class'] = rawData['Class']

elif dataChoice == 'house-votes':
    print('Importing File: house-votes-84.csv')
    rawData = pd.read_csv('house-votes-84.csv')

    # no cleaning up needed for the house votes dataset
    data = rawData

elif dataChoice == 'breast-cancer':
    print('Importing File: breast-cancer-wisconsin.csv')
    rawData = pd.read_csv('breast-cancer-wisconsin.csv', na_values='?')

    # clean up breast-cancer dataset by removing nan rows, then applying the floor function to every item
    nonNanData = rawData.dropna()
    data = nonNanData.drop(['Class'], axis = 1).applymap(np.floor).applymap(int).applymap(str)
    data['Class'] = nonNanData['Class']

elif dataChoice == 'glass':
    print('Importing File: glass.csv')
    rawData = pd.read_csv('glass.csv')

    # clean up glass dataset by rounding to the hundredths place
    data = rawData.drop(['Class'], axis = 1).round(2).applymap(str)
    data['Class'] = rawData['Class']

elif dataChoice == 'soybean':
    print('Importing File: soybean.csv')
    rawData = pd.read_csv('soybean.csv')

    # No cleaning up soybean database as all values have been normallized
    data = rawData

# call the method that performs the creation of models, and evaluates them
pp.pprint(perform2by5FoldValidation(data= data, shuffling = shuffling))