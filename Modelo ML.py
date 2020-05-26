# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 15:23:48 2019

@author: dougl
"""

# Protótipo Comparação de diversos algoritmos de Machine Learning - Revisão 01
# Haste de Âncora de 1m
# Author: Douglas Contente Pimentel Barbosa - Jan/2019

#------------------------------------------------------------------------------
#Logistic Regression
#------------------------------------------------------------------------------
def LogisticRegression2(X_train, y_train, X_test, y_test):    
    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Calculando Desempenho
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = (2*precision*recall)/(precision + recall)
    
    #Usando o método K-Cross Validation com k = 10
    from sklearn.model_selection import cross_val_score
    X_completo = np.concatenate((X_train, X_test))
    y_completo = np.concatenate((y_train, y_test))
    accuracies = cross_val_score(estimator = classifier, X = X_completo, y = y_completo, cv = 10, n_jobs = 1)
    meanCrossValScores = accuracies.mean()

    #Salvando os resultados
    return(accuracy, precision, recall, f1score, y_pred, meanCrossValScores)
    
#------------------------------------------------------------------------------
#K-Nearest Neighbors (k-NN)
#------------------------------------------------------------------------------
def KNearestNeighbors2(X_train, y_train, X_test, y_test):  
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Calculando Desempenho
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = (2*precision*recall)/(precision + recall)
    
    #Usando o método K-Cross Validation com k = 10
    from sklearn.model_selection import cross_val_score
    X_completo = np.concatenate((X_train, X_test))
    y_completo = np.concatenate((y_train, y_test))
    accuracies = cross_val_score(estimator = classifier, X = X_completo, y = y_completo, cv = 10, n_jobs = 1)
    meanCrossValScores = accuracies.mean()

    #Salvando os resultados
    return(accuracy, precision, recall, f1score, y_pred, meanCrossValScores)

#------------------------------------------------------------------------------
#Suport Vector Machine (SVM)
#------------------------------------------------------------------------------
def SuportVectorMachine2(X_train, y_train, X_test, y_test): 
    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Calculando Desempenho
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = (2*precision*recall)/(precision + recall)
    
    #Usando o método K-Cross Validation com k = 10
    from sklearn.model_selection import cross_val_score
    X_completo = np.concatenate((X_train, X_test))
    y_completo = np.concatenate((y_train, y_test))
    accuracies = cross_val_score(estimator = classifier, X = X_completo, y = y_completo, cv = 10, n_jobs = 1)
    meanCrossValScores = accuracies.mean()

    #Salvando os resultados
    return(accuracy, precision, recall, f1score, y_pred, meanCrossValScores)

#------------------------------------------------------------------------------
#Naïve Bayes
#------------------------------------------------------------------------------
def NaiveBayes2(X_train, y_train, X_test, y_test): 
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Calculando Desempenho
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = (2*precision*recall)/(precision + recall)
    
    #Usando o método K-Cross Validation com k = 10
    from sklearn.model_selection import cross_val_score
    X_completo = np.concatenate((X_train, X_test))
    y_completo = np.concatenate((y_train, y_test))
    accuracies = cross_val_score(estimator = classifier, X = X_completo, y = y_completo, cv = 10, n_jobs = 1)
    meanCrossValScores = accuracies.mean()

    #Salvando os resultados
    return(accuracy, precision, recall, f1score, y_pred, meanCrossValScores)
    
#------------------------------------------------------------------------------
#Decision Tree
#------------------------------------------------------------------------------
def DecisionTree2(X_train, y_train, X_test, y_test): 
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Calculando Desempenho
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = (2*precision*recall)/(precision + recall)
    
    #Usando o método K-Cross Validation com k = 10
    from sklearn.model_selection import cross_val_score
    X_completo = np.concatenate((X_train, X_test))
    y_completo = np.concatenate((y_train, y_test))
    accuracies = cross_val_score(estimator = classifier, X = X_completo, y = y_completo, cv = 10, n_jobs = 1)
    meanCrossValScores = accuracies.mean()

    #Salvando os resultados
    return(accuracy, precision, recall, f1score, y_pred, meanCrossValScores)
    
#------------------------------------------------------------------------------
#Random Forest
#------------------------------------------------------------------------------
def RandomForest2(X_train, y_train, X_test, y_test): 
    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Calculando Desempenho
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = (2*precision*recall)/(precision + recall)
    
    #Usando o método K-Cross Validation com k = 10
    from sklearn.model_selection import cross_val_score
    X_completo = np.concatenate((X_train, X_test))
    y_completo = np.concatenate((y_train, y_test))
    accuracies = cross_val_score(estimator = classifier, X = X_completo, y = y_completo, cv = 10, n_jobs = 1)
    meanCrossValScores = accuracies.mean()

    #Salvando os resultados
    return(accuracy, precision, recall, f1score, y_pred, meanCrossValScores)
   
#------------------------------------------------------------------------------
#Feedforward Artificial Neural Network - FF-ANN
#------------------------------------------------------------------------------
def ArtificialNeuralNetwork2(X_train, y_train, X_test, y_test): 
    from keras.models import Sequential
    from keras.layers import Dense
    
    def CriaRNA():
        # Iniciando RNA
        classifier = Sequential()
        # Hidden Layer #1
        classifier.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu', input_dim = 1001))
        # Hidden Layer #2
        #classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
        # Neurônios de Saída
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
        # Compilando RNA
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return(classifier)
        
    # Aplicando os dados de treino a RNA
    classifier = CriaRNA()
    classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 50)
    
    # Fazendo previsões dos dados de Teste
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    y_pred = np.int64(y_pred)
    y_pred = y_pred.reshape((-1,))
    
    # Calculando Desempenho
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = (2*precision*recall)/(precision + recall)
    
    #Usando o método K-Cross Validation com k = 10
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score
    X_completo = np.concatenate((X_train, X_test))
    y_completo = np.concatenate((y_train, y_test))
    classifier2 = KerasClassifier(build_fn = CriaRNA, batch_size = 10, epochs = 50)
    accuracies = cross_val_score(estimator = classifier2, X = X_completo, y = y_completo, scoring = 'accuracy', cv = 10, n_jobs = 1)
    meanCrossValScores = accuracies.mean()

    #Salvando os resultados
    return(accuracy, precision, recall, f1score, y_pred, meanCrossValScores)
   
#------------------------------------------------------------------------------    
#Programa Principal
#------------------------------------------------------------------------------

# Importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Definindo parâmetros

QdeSim = 1
QdeMetodos = 7 # 5 desconsiderando random forests e naive bayes

Accuracy_LR = 0.0
Accuracy_KNN = 0.0
Accuracy_SVM = 0.0
Accuracy_NB = 0.0
Accuracy_DT = 0.0
Accuracy_RF = 0.0
Accuracy_ANN = 0.0

Precision_LR = 0.0
Precision_KNN = 0.0
Precision_SVM = 0.0
Precision_NB = 0.0
Precision_DT = 0.0
Precision_RF = 0.0
Precision_ANN = 0.0

Recall_LR = 0.0
Recall_KNN = 0.0
Recall_SVM = 0.0
Recall_NB = 0.0
Recall_DT = 0.0
Recall_RF = 0.0
Recall_ANN = 0.0

F1Score_LR = 0.0
F1Score_KNN = 0.0
F1Score_SVM = 0.0
F1Score_NB = 0.0
F1Score_DT = 0.0
F1Score_RF = 0.0
F1Score_ANN = 0.0

Accuracy_Conj = 0.0
Precision_Conj = 0.0
Recall_Conj = 0.0
F1Score_Conj = 0.0  

CV_Score_LR = 0.0
CV_Score_KNN = 0.0
CV_Score_SVM = 0.0
CV_Score_NB = 0.0
CV_Score_DT = 0.0
CV_Score_RF = 0.0
CV_Score_ANN = 0.0

# Importando base de dados
dataset = pd.read_excel('BD-Sim-HA-1m-MF-2Classes - TEMP.xlsx')
X = dataset.iloc[:,:1001].values
#y = dataset.iloc[:, 2003].values #Usar este se for BD Med
y = dataset.iloc[:, 1007].values #Usar este se for BD Sim

for semente in range(QdeSim):
    # Dividindo a base de dados em Treinamento e Teste
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = semente)
    
    #Definindo vetor Previsoes
    PrevConj = np.zeros(len(y_test))
    
    # Normalizando entradas
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #Testando os métodos de AI
    aLR, pLR, rLR, sLR, predLR, cvLR = LogisticRegression2(X_train, y_train, X_test, y_test)
    Accuracy_LR = Accuracy_LR + aLR
    Precision_LR = Precision_LR + pLR
    Recall_LR = Recall_LR + rLR
    F1Score_LR = F1Score_LR + sLR
    PrevConj = PrevConj + predLR
    CV_Score_LR = CV_Score_LR + cvLR
    
    aKNN, pKNN, rKNN, sKNN, predKNN, cvKNN = KNearestNeighbors2(X_train, y_train, X_test, y_test)
    Accuracy_KNN = Accuracy_KNN + aKNN
    Precision_KNN = Precision_KNN + pKNN
    Recall_KNN = Recall_KNN + rKNN
    F1Score_KNN = F1Score_KNN + sKNN     
    PrevConj = PrevConj + predKNN
    CV_Score_KNN = CV_Score_KNN + cvKNN
        
    aSVM, pSVM, rSVM, sSVM, predSVM, cvSVM = SuportVectorMachine2(X_train, y_train, X_test, y_test)
    Accuracy_SVM = Accuracy_SVM + aSVM
    Precision_SVM = Precision_SVM + pSVM
    Recall_SVM = Recall_SVM + rSVM
    F1Score_SVM = F1Score_SVM + sSVM
    PrevConj = PrevConj + predSVM
    CV_Score_SVM = CV_Score_SVM + cvSVM
        
    aNB, pNB, rNB, sNB, predNB, cvNB = NaiveBayes2(X_train, y_train, X_test, y_test)
    Accuracy_NB = Accuracy_NB + aNB
    Precision_NB = Precision_NB + pNB
    Recall_NB = Recall_NB + rNB
    F1Score_NB = F1Score_NB + sNB
    PrevConj = PrevConj + predNB
    CV_Score_NB = CV_Score_NB + cvNB
        
    aDT, pDT, rDT, sDT, predDT, cvDT = DecisionTree2(X_train, y_train, X_test, y_test)
    Accuracy_DT = Accuracy_DT + aDT
    Precision_DT = Precision_DT + pDT
    Recall_DT = Recall_DT + rDT
    F1Score_DT = F1Score_DT + sDT
    PrevConj = PrevConj + predDT
    CV_Score_DT = CV_Score_DT + cvDT
        
    aRF, pRF, rRF, sRF, predRF, cvRF = RandomForest2(X_train, y_train, X_test, y_test)
    Accuracy_RF = Accuracy_RF + aRF
    Precision_RF = Precision_RF + pRF
    Recall_RF = Recall_RF + rRF
    F1Score_RF = F1Score_RF + sRF
    PrevConj = PrevConj + predRF
    CV_Score_RF = CV_Score_RF + cvRF
        
    aANN, pANN, rANN, sANN, predANN, cvANN = ArtificialNeuralNetwork2(X_train, y_train, X_test, y_test)
    Accuracy_ANN = Accuracy_ANN + aANN
    Precision_ANN = Precision_ANN + pANN
    Recall_ANN = Recall_ANN + rANN
    F1Score_ANN = F1Score_ANN + sANN
    PrevConj = PrevConj + predANN
    CV_Score_ANN = CV_Score_ANN + cvANN
       
    #Verificando o sistema de votação
    PrevConj = PrevConj/QdeMetodos
    PrevConj = (PrevConj > 0.5)
    PrevConj = np.int64(PrevConj)
    PrevConj = PrevConj.reshape((-1,))

    # Calculando a Matriz de Confusão para o sistema de votação
    # Calculando Desempenho
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    cm = confusion_matrix(y_test, PrevConj)
    #Salvando o resultado
    aConj = accuracy_score(y_test, PrevConj)
    pConj = precision_score(y_test, PrevConj)
    rConj = recall_score(y_test, PrevConj)
    f1sConj = (2*pConj*rConj)/(pConj + rConj)

    Accuracy_Conj = Accuracy_Conj + aConj
    Precision_Conj = Precision_Conj + pConj
    Recall_Conj = Recall_Conj + rConj
    F1Score_Conj = F1Score_Conj + f1sConj     
    
Accuracy_LR = Accuracy_LR / QdeSim
Accuracy_KNN = Accuracy_KNN / QdeSim
Accuracy_SVM = Accuracy_SVM / QdeSim
Accuracy_NB = Accuracy_NB / QdeSim
Accuracy_DT = Accuracy_DT / QdeSim
Accuracy_RF = Accuracy_RF / QdeSim
Accuracy_ANN = Accuracy_ANN / QdeSim

Precision_LR = Precision_LR / QdeSim
Precision_KNN = Precision_KNN / QdeSim
Precision_SVM = Precision_SVM / QdeSim
Precision_NB = Precision_NB / QdeSim
Precision_DT = Precision_DT / QdeSim
Precision_RF = Precision_RF / QdeSim
Precision_ANN = Precision_ANN / QdeSim

Recall_LR = Recall_LR / QdeSim
Recall_KNN = Recall_KNN / QdeSim
Recall_SVM = Recall_SVM / QdeSim
Recall_NB = Recall_NB / QdeSim
Recall_DT = Recall_DT / QdeSim
Recall_RF = Recall_RF / QdeSim
Recall_ANN = Recall_ANN / QdeSim

F1Score_LR = F1Score_LR / QdeSim
F1Score_KNN = F1Score_KNN / QdeSim
F1Score_SVM = F1Score_SVM / QdeSim
F1Score_NB = F1Score_NB / QdeSim
F1Score_DT = F1Score_DT / QdeSim
F1Score_RF = F1Score_RF / QdeSim
F1Score_ANN = F1Score_ANN / QdeSim

Accuracy_Conj = Accuracy_Conj / QdeSim
Precision_Conj = Precision_Conj / QdeSim  
Recall_Conj = Recall_Conj / QdeSim  
F1Score_Conj = F1Score_Conj / QdeSim    

CV_Score_LR = CV_Score_LR / QdeSim
CV_Score_KNN = CV_Score_KNN / QdeSim
CV_Score_SVM = CV_Score_SVM / QdeSim
CV_Score_NB = CV_Score_NB / QdeSim
CV_Score_DT = CV_Score_DT / QdeSim
CV_Score_RF = CV_Score_RF / QdeSim
CV_Score_ANN = CV_Score_ANN / QdeSim



    
   