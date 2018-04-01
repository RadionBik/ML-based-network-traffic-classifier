#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Radion Bikmukhamedov
"""
__version__ = '0.3'

from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn import metrics
import matplotlib.pyplot as plt
from time import time
import os
import configparser
import post_process_csv, live_mode, feature_extractor

def evaluationResults(ground_truth,predictions):
    '''
    evaluationResults() prints out Accuracy, F-measure (with average='weighted')
    scores

    In binary and multiclass classification, jaccard_similarity_score
    is equivalent to the ``accuracy_score``. It differs in the multilabel
    classification problem.
    '''
    #
    metricResult=metrics.jaccard_similarity_score(ground_truth,predictions)
    print('Accuracy on test set: %.4f' %metricResult)
    print('F-measure: %.4f' %metrics.f1_score(ground_truth,predictions,average='weighted'))
    #print(metrics.classification_report(ground_truth,predictions))
    return metricResult


def report(results, n_top=1):
    '''
    report() is a utility function to report best scores.
    not used for now
    '''
    for mean, std, params in zip(results['mean_test_score'], results['std_test_score'], results['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            if i==1:
                bestOne=results['params'][candidate]
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    return bestOne

def paramSearch(classifier,classifierName,X,y,config):

    '''
    paramSearch() returns the best parameters (estimated on 25% of the training
    set) for the classifiers, taking the grid values as the input for CV

    '''
    if __name__ == "__main__":
        numberOfSplitsCV=int(config['OfflineMode']['numberOfSplitsCV'])
        randomState=int(config['OfflineMode']['randomSeed'])

        param_dist = {
            'Linear':{"tol": [0.001,0.01],
                      #"max_features": sp_randint(1, 11),
                        'kernel': ['linear']},
            'LogRegr':{"C":[0.1,1,10,100,1000],
                      "tol": [0.00001,0.0001,0.001,0.01],
                      #"max_features": sp_randint(1, 11),
                        },
            'SVM' : [{'kernel': ['rbf'], 'gamma': [10,1,1e-2,1e-3,1e-4],
                         'C': [0.01, 0.1, 1, 10, 100, 100]},
                        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}],
            'DecTree': {"max_depth": [i for i in range(5,20) if i%3==1],
                      "max_features": [i for i in range(1,11) if i%2==1],
                      "criterion": ["gini","entropy"]},
            'RandomForest':{"n_estimators":[i for i in range(10,100) if i%10==1],
                      "max_depth": [i for i in range(5,20) if i%2==1],
    #                  #"max_features": sp_randint(1, 11),
                      "criterion": ["gini"]},
            'GradBoost':{"n_estimators" : [100],
                      "max_depth":  [i for i in range(1,6)],#sp_randint(1,6),
    #                  #"max_features": sp_randint(1, 11),
                      "learning_rate": [0.1,0.5,1]},
            'MLP':{"hidden_layer_sizes":([i for i in range(20,100) if i%20==1],[i for i in range(20,100) if i%20==1]),
                   "alpha":[0.0001,0.001,0.01,0.1,1,10]}
            }

        search = GridSearchCV(classifier, param_grid=param_dist[classifierName],n_jobs=-1,
                              scoring=make_scorer(metrics.jaccard_similarity_score),
                              cv=StratifiedKFold(n_splits=numberOfSplitsCV,
                              shuffle=True,random_state=randomState))
        start = time()
        search.fit(X, y)
        print("Search took %.2f seconds" %(time() - start))
        #   " parameter settings." % ((time() - start),sum([len(i) for i in param_dist[classifier].values()])))
        print('Best parameters are {} with score {:.4f}'.format(search.best_params_,
                                                                search.best_score_))
        #result = report(search.cv_results_,1)
        return dict(search.best_params_, **{'random_state':randomState})

def plotPrecision(results,config,appendix):
    '''
    plotPrecision() plots precision of tested ML alg-s and saves the plot
    to a pdf file
    '''
    y_pos=range(len(results))
    #plt.close()
    plt.figure()
    #sameNetTuple=sorted(net.items())
    plt.bar(y_pos,results.values(),width=0.5)
    plt.xticks(y_pos,results.keys(),rotation=45)
    plt.grid()
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.ylim(0.5,1)
    pathToSave=config['OfflineMode']['folderWithPlots']
    if config['OfflineMode'].getboolean('savePlotsToFile'):
        plt.savefig(os.pardir+os.sep+pathToSave+os.sep+'precision'+appendix+'.pdf')
    #plt.bar(zip(*sameNetTuple))

def plotConfusionMatrix(ground_truth,predictions,classes,classifName,config,appendix):
    """
    plot_confusion_matrix() plots the confusion matrix.
    Normalization can be applied by setting `normalizeConfusionMatrix=True`.
    """
    cm=confusion_matrix(ground_truth,predictions)
    normalize=config['OfflineMode'].getboolean('normalizeConfusionMatrix')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix:")
    else:
        print('Confusion matrix, without normalization:')

    print(pd.DataFrame(cm,classes,classes))

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('CM of {} classifier'.format(classifName))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #save to a pdf file
    pathToSave=config['OfflineMode']['folderWithPlots']
    plt.tight_layout()
    if config['OfflineMode'].getboolean('savePlotsToFile'):
        plt.savefig(os.pardir+os.sep+pathToSave+os.sep+'CM_of_'+classifName+appendix+'.pdf')
    #plt.close()

def offlineClassification(config):

    randomState=int(config['OfflineMode']['randomSeed'])

    useSeparateFileForTesting=config['OfflineMode'].getboolean('useSeparateFileForTesting')
    if useSeparateFileForTesting:
        appendix='_diffNet'
    else:
        appendix='_sameNet'

    pathToClassifiers=config['GeneralSettings']['folderWithTrainedClassifiers']
    useTrainedClasiffiers=config['OfflineMode'].getboolean('useTrainedClasiffiers')

    classifiersToTestList={
            'Linear':config['MLtoTest'].getboolean('Linear'),
            'LogRegr':config['MLtoTest'].getboolean('LogRegr'),
            'SVM' : config['MLtoTest'].getboolean('SVM'),
            'DecTree':config['MLtoTest'].getboolean('DecTree'),
            'RandomForest':config['MLtoTest'].getboolean('RandomForest'),
            'GradBoost':config['MLtoTest'].getboolean('GradBoost'),
            'MLP':config['MLtoTest'].getboolean('MLP')
            }

    classifiersToOptimizeList={
            'Linear':config['MLtoOptimize'].getboolean('Linear'),
            'LogRegr':config['MLtoOptimize'].getboolean('LogRegr'),
            'SVM' : config['MLtoOptimize'].getboolean('SVM'),
            'DecTree':config['MLtoOptimize'].getboolean('DecTree'),
            'RandomForest':config['MLtoOptimize'].getboolean('RandomForest'),
            'GradBoost':config['MLtoOptimize'].getboolean('GradBoost'),
            'MLP':config['MLtoOptimize'].getboolean('MLP')
            }

    #########################################################################
    #initialize, train and optimize classifiers
    #c=100000 is large enough for logRegr to be considered as a linear algorithm
    classifiersDefaults={
        'Linear':LogisticRegression(C=100000,random_state=randomState),
        'LogRegr':LogisticRegression(random_state=randomState),
        'SVM' : SVC(random_state=randomState),
        'DecTree':DecisionTreeClassifier(random_state=randomState),
        'RandomForest':RandomForestClassifier(random_state=randomState),
        'GradBoost':GradientBoostingClassifier(random_state=randomState),
        'MLP':MLPClassifier(random_state=randomState)
        }

    #get features and labels from .csv file(s)
    featuresTrain,featuresTest,labelsTrain,labelsTest,le=post_process_csv.getFeaturesAndLabelsFromCsv(config)

    testResults=dict()
    plt.ioff()
    for classifierName in [i for i in list(classifiersToTestList.keys()) if classifiersToTestList[i]]:
        print('=======================\n{} classifier:'.format(classifierName))
        if not useTrainedClasiffiers:
            #optimize classifiers
            if classifiersToOptimizeList[classifierName]:
                optimalParameters=paramSearch(classifiersDefaults[classifierName],
                                              classifierName,featuresTrain,
                                              labelsTrain,config)
                classifier=classifiersDefaults[classifierName].set_params(**optimalParameters)
            else:
                classifier=classifiersDefaults[classifierName]

            #fit classifier
            startTime=time()
            classifier.fit(featuresTrain,labelsTrain)
            print('Fitting time for {} is {:.3f} s'.format(str(classifier).split('(')[0],time()-startTime))
            #save the classifier to a file
            joblib.dump(classifier, os.pardir+os.sep+pathToClassifiers+os.sep+
                         classifierName+'.cla')
        else:
            classifier=joblib.load(os.pardir+os.sep+pathToClassifiers+os.sep+
                                   classifierName+'.cla')
        #####################################################################
        #test classifier
        labelsTestPredicted = classifier.predict(featuresTest)
        #get the accuracy of the classifier
        results=evaluationResults(labelsTest,labelsTestPredicted)
        testResults.update({classifierName:results})
        
        #####################################################################
        #reporting part
        font = {#'family' : 'normal',
            #'weight' : 'bold',
            'size'   : 12}
        plt.rc('font', **font)
        #optionally draw CM
        if config['OfflineMode'].getboolean('plotConfusionMatrix'):
            plotConfusionMatrix(labelsTest,labelsTestPredicted,le.classes_,
                                classifierName,config,appendix)
    #optionally plot accuracy scores
    if config['OfflineMode'].getboolean('plotAccuracyScores'):
        plotPrecision(testResults,config,appendix)

def onlineClassification(config):

    #packetLimit = 5
    if config['OnlineMode'].getboolean('selectDevice'):
        targetDevice = live_mode.selectDevice()
    else:
        targetDevice = config['OnlineMode']['selectedDevice']
    #empty dict for flows, outside of captureFlows() to preserve the values
    flows = {}
    #define the desired classifier
    onlineClassifier = config['OnlineMode']['onlineClassifier']
    numberOfPacketsToWait = int(config['OnlineMode']['numberOfPacketsToWait'])

    print('{} has been selected, listening on {}'.format(onlineClassifier,targetDevice))

    pathToClassifiers = config['GeneralSettings']['folderWithTrainedClassifiers']
    try: classifier = joblib.load(os.pardir+os.sep+pathToClassifiers+os.sep+
                                   onlineClassifier+'.cla')
    except: print('Cannot read the file with the classifier')
    try: scaler = joblib.load(os.pardir+os.sep+pathToClassifiers+os.sep+'scaler.dat')
    except: print('Cannot read the file with the scaler')
    try: le = joblib.load(os.pardir+os.sep+pathToClassifiers+os.sep+'le.dat')
    except: print('Cannot read the file with the Label Encoder')

    #iterate infinitely to capture the flows 
    while True:
        key,flow = next(live_mode.liveFlowCaptureDpkt(targetDevice,flows,numberOfPacketsToWait))
        stats = feature_extractor.getFlowStatsDpkt(flow)

        #convert to DataFrame with features as the header
        #flowFeatures = pd.DataFrame.from_dict(stats,orient="index").T
        flowFeatures = live_mode.statsToDataFrame(stats)
        for i in range(5):
            flowFeatures.to_csv(os.pardir+os.sep+config['OfflineMode']['folderWithCSVfiles']+os.sep+'flow'+str(i)+'.csv', index=False)
        #scale the features
        features=scaler.transform(flowFeatures)
        predictedLabel = classifier.predict(features)

        print(key,' -- ',le.classes_[predictedLabel[0]])

def main():
    #########################################################################
    #read the config parameters from the file
    config = configparser.ConfigParser()
    config.read(os.pardir+os.sep+'config.ini')
    if config['GeneralSettings'].getboolean('modeIsOnline'):
        onlineClassification(config)
    else:
        offlineClassification(config)



if __name__ == "__main__":
    main()
