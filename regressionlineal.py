# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el Data Set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0 )


#Crear modelo de regresion lineal simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto del test
y_pred = regression.predict(X_test)

#Visualizar los resultados del entrenamiento
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color ="blue")
plt.title("Sueldo vs Años de EXperiencia")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo en $USD")
plt.show()
#Visualizar los resultados del test
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regression.predict(X_train), color ="blue")
plt.title("Sueldo vs Años de EXperiencia conjunto de TEsting")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo en $USD")
plt.show()