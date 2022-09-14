# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:34:51 2020
@author:ADRIAN MONTERO
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#IMPORTANDO DATASET Y GRAFICANDO PUNTOS DEL ARCHIVO .CSV(GRAFICA 1)  
df = pd.read_csv("./china_gdp.csv")
df.head(10)
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#DEFINIMOS FUNCION SIGMOIDE INICIALMENTE CON SUS PARAMETROS BETA
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

beta_1 = 0.40
beta_2 = 1990.0

#función logística
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#predicción de puntos
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')
# Normalicemos nuestros datos
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

#OPTIMIZACION DE PARAMETROS BETA
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)

#imprimir los parámetros finales
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()