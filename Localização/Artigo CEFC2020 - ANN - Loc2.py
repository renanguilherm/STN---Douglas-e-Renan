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
#Programa para localização da falha
#------------------------------------------------------------------------------

#Define parâmetro a ser analisado
X = ReZin

Resultados = np.array(['Med', 'DesvPad', 'Min', 'Max'], ndmin = 2)
X_train, X_test, y_train, y_test, entradasRNA = Trata_Dados(X, y, variance = 0.99)
Previsoes = y_test # A primeira coluna é o "gabarito"

print(entradasRNA)

ValTest = [int(round((entradasRNA/3),0)), int(round((entradasRNA/2),0)),
           entradasRNA, 2*entradasRNA, 3*entradasRNA]

ArqTest = [[ValTest[0]], [ValTest[1]], [ValTest[2]], [ValTest[3]], [ValTest[4]],
           [ValTest[0], ValTest[0]], [ValTest[1], ValTest[1]], [ValTest[2], ValTest[2]],
           [ValTest[3], ValTest[3]], [ValTest[4], ValTest[4]],
           [ValTest[0], ValTest[0], ValTest[0]], [ValTest[1], ValTest[1], ValTest[1]],
           [ValTest[2], ValTest[2], ValTest[2]], [ValTest[3], ValTest[3], ValTest[3]],
           [ValTest[4], ValTest[4], ValTest[4]]]

#Realiza as simulações com as diversas arquiteturas
for i in range(len(ArqTest)):
    RNA = Treina_RNA(X_train, y_train, n_entradas=entradasRNA,
                     Neuronios_por_camada = ArqTest[i],
                     epocas=200, Reg = 0.001)
    y_pred, Erro_Abs, Max, Min, Media, DesvPad = Calc_Estats(RNA, X_test, y_test)
    #Plota_Treino(RNA)
    Apresenta_Resultado(Max, Min, Media, DesvPad)
    Scores = np.array([round(Media, 3), round(DesvPad, 3), round(Min, 3), round(Max, 3)], ndmin = 2)
    
    Resultados = np.concatenate((Resultados, Scores), axis = 0)
    Previsoes = np.concatenate((Previsoes, y_pred), axis = 1)
#Salva os resultados em um arquivo excel
    Salva_Excel('Resultados', 'ImZin', Resultados)
    Salva_Excel('Previsoes', 'ImZin', Previsoes)

#Envia email para informar que a simulação foi concluída
Envia_Email()
