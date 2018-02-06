# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:05:22 2018

@author: Radion Bikmukhamedov
"""
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def cleanUpCSVfile(filename,thresholdForFlowDeleting,config):
    '''
    cleanUpCSVfile() loads a .csv file from the "training_data" folder
    with features and labels and removes rare protocols and flows
    '''
    trainingDataFolderName=config['GeneralSettings']['folderWithCSVfiles']
    data = pd.read_csv(os.pardir+os.sep+trainingDataFolderName+os.sep+filename)
    #print(data.proto.value_counts())

    #convert SSL_No_cert to SSL
    data.replace('SSL_No_Cert','SSL',inplace=True)
    #delete Unknown, NaNs in 'proto' and fill in NaNs with 0
    data=data[data.proto!='Unknown']
    data.dropna(subset=['proto'],inplace=True)
    data.drop('subproto',axis=1,inplace=True)
    data.fillna(0,inplace=True)

    #delete rarely occuring flows, identifying them first
    indexToDel=[]
    for i in range(len(data.proto.value_counts())):
        if data.proto.value_counts()[i]<thresholdForFlowDeleting:
            indexToDel.append(data.proto.value_counts().index[i])

    deleteProtos(data,indexToDel)
    return data


def deleteProtos(data,protosToDel):
    '''
    deleteProtos() deletes seldom occuring protocols, takes DataFrame
    with features and list with indexes of undesired protos as arguments
    '''
    for i in protosToDel:
        data.drop(data[data.proto==i].index,axis=0,inplace=True)


def getTrainFeaturesAndLabels(data):
    '''
    getTrainFeaturesAndLabels() scales features and encodes labels using 
    globally defined variables of the LabelEncoder() and StandardScaler() 
    objects
    '''
    #set up scaler and encoder
    le=LabelEncoder()
    scaler=StandardScaler()
    le.fit(list(set(data.proto)))
    scaler.fit(data.drop('proto',axis=1) )
    #scale feature and encode labels
    features=scaler.transform(data.drop('proto',axis=1))
    target=le.transform(data.proto)
    return features,target,le,scaler

def getTestFeaturesAndLabels(data,le,scaler):
    '''
    getTestFeaturesAndLabels() scales features and encodes labels using 
    obtained from
    getTrainFeaturesAndLabels() objects
    '''
    features=scaler.transform(data.drop('proto',axis=1))
    target=le.transform(data.proto)
    return features, target

def getFeaturesAndLabelsFromCsv(config):
    '''
    getFeaturesAndLabelsFromCsv() reads, cleans up and returns features 
    with labels from .csv files
    '''
    randomState=int(config['GeneralSettings']['randomSeed'])
    useSeparateFileForTesting=config['GeneralSettings'].getboolean('useSeparateFileForTesting')
    #names of .csv files with features and labels
    dumpTrain=config['GeneralSettings']['fileForTraining']
    dumpTest=config['GeneralSettings']['fileForTesting']

    #extact info from .csv files and clean them up
    thresholdForFlowDeleting=int(config['GeneralSettings']['thresholdForFlowDeleting'])
    if useSeparateFileForTesting:
        dataTrain=cleanUpCSVfile(dumpTrain,thresholdForFlowDeleting,config)
        dataTest=cleanUpCSVfile(dumpTest,thresholdForFlowDeleting,config)
    else:
        data=cleanUpCSVfile(dumpTrain,thresholdForFlowDeleting,config)
        dataTrain, dataTest = train_test_split(data,shuffle=True,test_size=0.3,random_state=randomState)

    #delete protos that are NOT in training AND test sets, leaving only the common
    diffProtos=list(set(dataTest.proto).symmetric_difference(set(dataTrain.proto)))
    deleteProtos(dataTest,diffProtos)
    deleteProtos(dataTrain,diffProtos)

    #obtain normalized train and test features/labels
    featuresTrain,labelsTrain,le,scaler = getTrainFeaturesAndLabels(dataTrain)
    featuresTest,labelsTest = getTestFeaturesAndLabels(dataTest,le,scaler)
    return featuresTrain,featuresTest,labelsTrain,labelsTest,le