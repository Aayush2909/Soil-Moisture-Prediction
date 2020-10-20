# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

from Utils import relationPlotter, historyPlotter

LEARNING_RATE = 0.001       #Learning Rate for the model
EPOCHS = 200        #Number of epochs
MOMENTUM = 0.9      #Momentum for better learning
BATCH_SIZE = 1024       #batch size

#Function to split data into two parts: Training and Testing
def splitData(X, Y, test_size=0.3):
    X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


#Model Building Function with given input dimensions
def buildModel(input_dim):
    model = Sequential()
    model.add(Dense(36, activation='linear', kernel_initializer='random_uniform', kernel_regularizer=l2(0.001), input_dim=input_dim))
    model.add(Dropout(0.1))
    model.add(Dense(18, activation='linear',kernel_regularizer=l2(0.01), kernel_initializer='random_uniform'))
    model.add(Dropout(0.1))
    model.add(Dense(9, activation='linear',kernel_regularizer=l2(0.01), kernel_initializer='random_uniform'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear', kernel_initializer='uniform'))

    return model
    

if __name__ == '__main__':

    #Loading Datasets
    field1 = pd.read_csv("Soil Dataset/TVC_logger_1_sedge.csv")
    field2 = pd.read_csv("Soil Dataset/TVC_logger_2_riparian.csv")
    field3 = pd.read_csv("Soil Dataset/TVC_logger_ 3_betula.csv")
    field4 = pd.read_csv("Soil Dataset/TVC_logger_4_alder.csv")

    print(field1.head())
    print(field2.head())
    print(field3.head())
    print(field4.head())
    print()


    #Considering important features only(Tsoil, Tair, RHpercent and Moisture)
    field1 = field1.iloc[:,2:]
    field1.rename(columns={'Tsoil.C': 'Tsoil', 'Tair.C': 'Tair', 'RH.percent':'RHpercent', 'VWC.m3.per.m3':'Soil Moisture'}, inplace=True)

    field2 = field2.iloc[:,2:-1]
    field2.rename(columns={'Tsoil.C.salix': 'Tsoil', 'Tair.C': 'Tair', 'RH.percent':'RHpercent', 'VWC.m3.per.m3.moss':'Soil Moisture'}, inplace=True)

    field3 = field3.iloc[:,2:]
    field3.rename(columns={'Tsoil.C.hummock': 'Tsoil', 'Tair.C': 'Tair', 'RH.percent':'RHpercent', 'VWC.m3.per.m3.hummock':'Soil Moisture'}, inplace=True)

    field4 = field4.iloc[:,2:-1]
    field4.rename(columns={'Tsoil.C.hummock': 'Tsoil', 'Tair.C': 'Tair', 'RH.percent':'RHpercent', 'VWC.m3.per.m3.hummock':'Soil Moisture'}, inplace=True)


    #Dropping rows having NaN values
    field1.dropna(inplace=True)
    field2.dropna(inplace=True)
    field3.dropna(inplace=True)
    field4.dropna(inplace=True)

    fields = [field1, field2, field3, field4]

    for n,f in enumerate(fields):
        relationPlotter(f, n+1)

    data = pd.concat(fields)        #Concatenating all field's data


    X = data.iloc[:,:3].values
    Y = data.iloc[:,3:].values
    X_train, X_test, Y_train, Y_test = splitData(X , Y, test_size=0.3)
    input_dim = X.shape[1]
    
    smp_model = buildModel(input_dim)
    print("A DL model of Soil Moisture Prediction-")
    print(smp_model.summary())          #model summary containing no. of neurons, layers and parameters.
    print()

    sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
    smp_model.compile(loss=MeanSquaredError(), optimizer=sgd)

    #Training
    print("Model training starts...\n")
    smp_model_history = smp_model.fit(X_train, Y_train,
                                    batch_size=BATCH_SIZE,
                                    epochs=EPOCHS,
                                    verbose=1,
                                    validation_data=(X_test, Y_test))
    print()

    historyPlotter(smp_model_history)

    sample_data = data.sample(n=10)

    inp = sample_data.iloc[:,:3].values
    target = sample_data.iloc[:,3:].values
    out = smp_model.predict(inp)


    #Prediction
    print("Predictions on some random data -\n")
    for i,o,t in zip(inp, out, target):
        print("Soil Temperature: {}°C".format(i[0]))
        print("Air Temperature: {}°C".format(i[1]))
        print("Relative Humidity Percent: {}%".format(i[2]))
        print("Expected Soil Moisture: {}".format(t[0]))
        print("Predicted Soil Moisture: {}".format(o[0]))
        print()


    
    