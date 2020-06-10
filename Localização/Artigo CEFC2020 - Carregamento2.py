# -*- coding: utf-8 -*-
"""
Created on Sat Aug 03 22:23:48 2019

@author: dougl
"""

# Protótipo RNA Haste de Âncora Simulações - Revisão 02
#Localizção
# Haste de Âncora de 01m
# Base de Dados: Simulações HFSS
# Author: Douglas Contente Pimentel Barbosa - Ago/2019

#------------------------------------------------------------------------------    
#Programa para carregamento da base de dados e definição das funções
#------------------------------------------------------------------------------

# Importando bibliotecas
from datetime import datetime
instante_inicial = datetime.now()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import smtplib
import xlwt

# Importando base de dados
dataset = pd.read_excel('Base_7_parâmetros - Rotulada Localização - Sem Zero.xlsx')
ModS11 = dataset.iloc[:,0:1001].values
FasS11 = dataset.iloc[:,1001:2002].values
VSWR = dataset.iloc[:,2002:3003].values
ReS11 = dataset.iloc[:,3003:4004].values
ImS11 = dataset.iloc[:,4004:5005].values
ReZin = dataset.iloc[:,5005:6006].values
ImZin = dataset.iloc[:,6006:7007].values

#y = dataset.iloc[:,2008:2009].values #Usado para 2 parâmetros - Módulo e Fase
y = dataset.iloc[:,7011:7012].values
Freq = np.arange(2, 1000.1, 0.998)

#Definindo parâmetros auxiliares serem analisados
#Módulo e Fase concatenados
ModeFasS11 = np.concatenate((ModS11, FasS11), axis = 1)
#Módulo Normalizado
ModS11N = ModS11/Freq
#Fase Normalizada
FasS11N = FasS11/Freq
#Módulo e Fase Normalizado
ModeFasS11N = np.concatenate((ModS11N, FasS11N), axis = 1)

#FUNÇÕES
def Trata_Dados(X, y, variance = 1):
    # Dividindo a base de dados em Treinamento e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # Normalizando entradas
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Aplicando PCA
    entradasRNA = 0
    pca = PCA(n_components = None)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    variance = 0
    for element in (explained_variance):
        variance += element
        entradasRNA += 1
        if variance >= 0.99:
            break
    pca = PCA(n_components = entradasRNA)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return(X_train, X_test, y_train, y_test, entradasRNA)

def Cria_RNA(Neuronios_por_camada, n_entradas, Reg):
    #PCA:128 128-128-1 0.13/5.37
    #PCA:128 64-64-1 0.004/5.76
    #PCA:128 32-32-1  0.039/8.3
    # Iniciando RNA
    classifier = Sequential()
    if(len(Neuronios_por_camada) == 1):
        # Hidden Layer #1
        classifier.add(Dense(output_dim = Neuronios_por_camada[0], init = 'uniform', activation = 'relu', input_dim = n_entradas, kernel_regularizer=regularizers.l2(Reg)))
    elif(len(Neuronios_por_camada) == 2):
        # Hidden Layer #1
        classifier.add(Dense(output_dim = Neuronios_por_camada[0], init = 'uniform', activation = 'relu', input_dim = n_entradas, kernel_regularizer=regularizers.l2(Reg)))
        # Neurônios Camada Oculta
        classifier.add(Dense(output_dim = Neuronios_por_camada[1], init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(Reg)))
    elif(len(Neuronios_por_camada) == 3):
        # Hidden Layer #1
        classifier.add(Dense(output_dim = Neuronios_por_camada[0], init = 'uniform', activation = 'relu', input_dim = n_entradas, kernel_regularizer=regularizers.l2(Reg)))
        # Neurônios Camada Oculta
        classifier.add(Dense(output_dim = Neuronios_por_camada[1], init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(Reg)))
        # Neurônios Camada Oculta
        classifier.add(Dense(output_dim = Neuronios_por_camada[2], init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(Reg)))
    elif(len(Neuronios_por_camada) == 4):
        # Hidden Layer #1
        classifier.add(Dense(output_dim = Neuronios_por_camada[0], init = 'uniform', activation = 'relu', input_dim = n_entradas, kernel_regularizer=regularizers.l2(Reg)))
        # Neurônios Camada Oculta
        classifier.add(Dense(output_dim = Neuronios_por_camada[1], init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(Reg)))
        # Neurônios Camada Oculta
        classifier.add(Dense(output_dim = Neuronios_por_camada[2], init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(Reg)))
        # Neurônios Camada Oculta
        classifier.add(Dense(output_dim = Neuronios_por_camada[3], init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(Reg)))
    elif(len(Neuronios_por_camada) == 5):
        # Hidden Layer #1
        classifier.add(Dense(output_dim = Neuronios_por_camada[0], init = 'uniform', activation = 'relu', input_dim = n_entradas, kernel_regularizer=regularizers.l2(Reg)))
        # Neurônios Camada Oculta
        classifier.add(Dense(output_dim = Neuronios_por_camada[1], init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(Reg)))
        # Neurônios Camada Oculta
        classifier.add(Dense(output_dim = Neuronios_por_camada[2], init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(Reg)))
        # Neurônios Camada Oculta
        classifier.add(Dense(output_dim = Neuronios_por_camada[3], init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(Reg)))
        # Neurônios Camada Oculta
        classifier.add(Dense(output_dim = Neuronios_por_camada[4], init = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(Reg)))
    else:
        print("Problema no múmero de camadas")
        return
    # Neurônios de Saída
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu'))
    # Compilando RNA
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    return(classifier)

def Treina_RNA(X_train, y_train, n_entradas, Neuronios_por_camada, epocas, Reg):
    # Aplicando os dados de treino a RNA
    RNA = Cria_RNA(Neuronios_por_camada, n_entradas, Reg)
    RNA.fit(X_train, y_train, batch_size = 10, nb_epoch = epocas)
    return(RNA)    

def Calc_Estats(RNA, X_test, y_test):
    y_pred = RNA.predict(X_test)
    Erro_Abs = y_pred - y_test
    Max = Erro_Abs.max()
    Min = Erro_Abs.min()
    Media = Erro_Abs.mean()
    DesvPad = Erro_Abs.std()
    return(y_pred, Erro_Abs, Max, Min, Media, DesvPad)

def Plota_Treino(RNA):
#    Fig1 = plt.figure(1)
    plt.clf()
    plt.plot(RNA.history['loss'])
    plt.title('Convergência - Treinamento')
    plt.ylabel('Erro')
    plt.xlabel('Épocas')
    plt.show()

def Apresenta_Resultado(Max, Min, Media, DesvPad):
    #Apresentando resultados
    print("Max e Min = " + str(round(Min, 2)) + " a " + str(round(Max, 2)))
    print("Conf.Int.95% = " + str(round((Media - 2*DesvPad), 2)) + " > " + str(round(Media, 2)) + " > " + str(round((Media + 2*DesvPad), 2)) + " Range: " + str(round(4*DesvPad)))    

#Enviando email de aviso
def Envia_Email(Mensagem = ''):
    smtp = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    
    smtp.login('zezinhoguarah@gmail.com', 'ljgyvoezdxfliycq')
    
    de = 'zezinhoguarah@gmail.com'
    para = ['douglas.contente@gmail.com']
    msg = """From: %s
    To: %s
    Subject: Avanco simulacao STN (TESTE)
    
    Simulacao concluida no PC STN01.""" % (de, ', '.join(para))
    
    smtp.sendmail(de, para, msg)
    
    smtp.quit()
    return

#Salvando resultados
def Salva_Excel(NomeArquivo, Parametro, NP_Array):
    wb = xlwt.Workbook()
    ws = wb.add_sheet(Parametro) #Nome da aba
    for i in range(NP_Array.shape[0]):
        for j in range(NP_Array.shape[1]):
            ws.write(i, j, NP_Array[i][j])
#    
#    ws.write(0, 0, 1234.56, style0)
#    ws.write(1, 0, datetime.now(), style1)
#    ws.write(2, 0, 1)
#    ws.write(2, 1, 2)
#    ws.write(2, 2, 3)
#    ws.write(2, 3, xlwt.Formula("A3+B3"))
    
    wb.save(NomeArquivo +'.xls') #nome do arquivo

def Divide10Conjuntos(X, y):
    XTest = []
    XTrain = []
    yTest = []
    yTrain = []
    
    xX50_1, xX50_2, yy50_1, yy50_2 = train_test_split(X, y, 
                                                        test_size = 0.5, 
                                                        random_state = 0)
    xX40_1, xX10_1, yy40_1, yy10_1 = train_test_split(xX50_1, yy50_1, 
                                                        test_size = 0.2, 
                                                        random_state = 0)    
    xX40_2, xX10_2, yy40_2, yy10_2 = train_test_split(xX50_2, yy50_2, 
                                                        test_size = 0.2, 
                                                        random_state = 0)
    xX20_1, xX20_2, yy20_1, yy20_2 = train_test_split(xX40_1, yy40_1, 
                                                        test_size = 0.5, 
                                                        random_state = 0)    
    xX20_3, xX20_4, yy20_3, yy20_4 = train_test_split(xX40_2, yy40_2, 
                                                        test_size = 0.5, 
                                                        random_state = 0)
    xX10_3, xX10_4, yy10_3, yy10_4 = train_test_split(xX20_1, yy20_1, 
                                                        test_size = 0.5, 
                                                        random_state = 0)
    xX10_5, xX10_6, yy10_5, yy10_6 = train_test_split(xX20_2, yy20_2, 
                                                        test_size = 0.5, 
                                                        random_state = 0)
    xX10_7, xX10_8, yy10_7, yy10_8 = train_test_split(xX20_3, yy20_3, 
                                                        test_size = 0.5, 
                                                        random_state = 0)
    xX10_9, xX10_0, yy10_9, yy10_0 = train_test_split(xX20_4, yy20_4, 
                                                        test_size = 0.5, 
                                                        random_state = 0)
    xTrn0 = np.concatenate((xX10_1, xX10_2, xX10_3, xX10_4, 
                            xX10_5, xX10_6, xX10_7, xX10_8, xX10_9), axis = 0)
    yTrn0 = np.concatenate((yy10_1, yy10_2, yy10_3, yy10_4,
                            yy10_5, yy10_6, yy10_7, yy10_8, yy10_9), axis = 0)
    xTst0 = xX10_0
    yTst0 = yy10_0
    
    XTest.append(xTst0)
    XTrain.append(xTrn0)
    yTest.append(yTst0)
    yTrain.append(yTrn0)    
    
    xTrn1 = np.concatenate((xX10_0, xX10_2, xX10_3, xX10_4, 
                            xX10_5, xX10_6, xX10_7, xX10_8, xX10_9), axis = 0)
    yTrn1 = np.concatenate((yy10_0, yy10_2, yy10_3, yy10_4,
                            yy10_5, yy10_6, yy10_7, yy10_8, yy10_9), axis = 0)
    xTst1 = xX10_1
    yTst1 = yy10_1
    
    XTest.append(xTst1)
    XTrain.append(xTrn1)
    yTest.append(yTst1)
    yTrain.append(yTrn1)    
        
    xTrn2 = np.concatenate((xX10_1, xX10_0, xX10_3, xX10_4, 
                            xX10_5, xX10_6, xX10_7, xX10_8, xX10_9), axis = 0)
    yTrn2 = np.concatenate((yy10_1, yy10_0, yy10_3, yy10_4,
                            yy10_5, yy10_6, yy10_7, yy10_8, yy10_9), axis = 0)
    xTst2 = xX10_2
    yTst2 = yy10_2
    
    XTest.append(xTst2)
    XTrain.append(xTrn2)
    yTest.append(yTst2)
    yTrain.append(yTrn2)    
        
    xTrn3 = np.concatenate((xX10_1, xX10_2, xX10_0, xX10_4, 
                            xX10_5, xX10_6, xX10_7, xX10_8, xX10_9), axis = 0)
    yTrn3 = np.concatenate((yy10_1, yy10_2, yy10_0, yy10_4,
                            yy10_5, yy10_6, yy10_7, yy10_8, yy10_9), axis = 0)
    xTst3 = xX10_3
    yTst3 = yy10_3
    
    XTest.append(xTst3)
    XTrain.append(xTrn3)
    yTest.append(yTst3)
    yTrain.append(yTrn3)    
        
    xTrn4 = np.concatenate((xX10_1, xX10_2, xX10_3, xX10_0, 
                            xX10_5, xX10_6, xX10_7, xX10_8, xX10_9), axis = 0)
    yTrn4 = np.concatenate((yy10_1, yy10_2, yy10_3, yy10_0,
                            yy10_5, yy10_6, yy10_7, yy10_8, yy10_9), axis = 0)
    xTst4 = xX10_4
    yTst4 = yy10_4
    
    XTest.append(xTst4)
    XTrain.append(xTrn4)
    yTest.append(yTst4)
    yTrain.append(yTrn4)    
        
    xTrn5 = np.concatenate((xX10_1, xX10_2, xX10_3, xX10_4, 
                            xX10_0, xX10_6, xX10_7, xX10_8, xX10_9), axis = 0)
    yTrn5 = np.concatenate((yy10_1, yy10_2, yy10_3, yy10_4,
                            yy10_0, yy10_6, yy10_7, yy10_8, yy10_9), axis = 0)
    xTst5 = xX10_5
    yTst5 = yy10_5
    
    XTest.append(xTst5)
    XTrain.append(xTrn5)
    yTest.append(yTst5)
    yTrain.append(yTrn5)    
        
    xTrn6 = np.concatenate((xX10_1, xX10_2, xX10_3, xX10_4, 
                            xX10_5, xX10_0, xX10_7, xX10_8, xX10_9), axis = 0)
    yTrn6 = np.concatenate((yy10_1, yy10_2, yy10_3, yy10_4,
                            yy10_5, yy10_0, yy10_7, yy10_8, yy10_9), axis = 0)
    xTst6 = xX10_6
    yTst6 = yy10_6
    
    XTest.append(xTst6)
    XTrain.append(xTrn6)
    yTest.append(yTst6)
    yTrain.append(yTrn6)    
        
    xTrn7 = np.concatenate((xX10_1, xX10_2, xX10_3, xX10_4, 
                            xX10_5, xX10_6, xX10_0, xX10_8, xX10_9), axis = 0)
    yTrn7 = np.concatenate((yy10_1, yy10_2, yy10_3, yy10_4,
                            yy10_5, yy10_6, yy10_0, yy10_8, yy10_9), axis = 0)
    xTst7 = xX10_7
    yTst7 = yy10_7
    
    XTest.append(xTst7)
    XTrain.append(xTrn7)
    yTest.append(yTst7)
    yTrain.append(yTrn7)    
        
    xTrn8 = np.concatenate((xX10_1, xX10_2, xX10_3, xX10_4, 
                            xX10_5, xX10_6, xX10_7, xX10_0, xX10_9), axis = 0)
    yTrn8 = np.concatenate((yy10_1, yy10_2, yy10_3, yy10_4,
                            yy10_5, yy10_6, yy10_7, yy10_0, yy10_9), axis = 0)
    xTst8 = xX10_8
    yTst8 = yy10_8
    
    XTest.append(xTst8)
    XTrain.append(xTrn8)
    yTest.append(yTst8)
    yTrain.append(yTrn8)    
        
    xTrn9 = np.concatenate((xX10_1, xX10_2, xX10_3, xX10_4, 
                            xX10_5, xX10_6, xX10_7, xX10_8, xX10_0), axis = 0)
    yTrn9 = np.concatenate((yy10_1, yy10_2, yy10_3, yy10_4,
                            yy10_5, yy10_6, yy10_7, yy10_8, yy10_0), axis = 0)
    xTst9 = xX10_9
    yTst9 = yy10_9
    
    XTest.append(xTst9)
    XTrain.append(xTrn9)
    yTest.append(yTst9)
    yTrain.append(yTrn9)    
        
    return(XTrain, XTest, yTrain, yTest)
    
instante_final = datetime.now()
tempo_carregamento1 = instante_final - instante_inicial    
'----------------------------------------------------------------------------'
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
#
#title = "Learning Curve (ANN)"
## SVC is more expensive so we do a lower number of CV iterations:
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#estimator = classifier
#plot_learning_curve(estimator, title, X_train, y_train, (0.7, 1.01), cv=cv, n_jobs=1)
#
#plt.show()