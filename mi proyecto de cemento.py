# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:27:56 2023

@author: Mikel
"""
#En este proyecto vamos a analizar una serie de datos sobre la composicion de varias mezclas de hormigon y analizar la variable objetivo que es su fuerza(strength). Por lo
#tanto estamos ante un problema de regresion lineal ya que la variable a analizar es continua.

# Requerimientos para realizar un modelo supervisado
#No missing values
#Data in numeric format
#Data stored in pandas Dataframe
#Perform Exploratory data Analysis (EDA)
#Cargamos los datos
import os
import pandas as pd
os.chdir(r"C:\Users\PCUser\Desktop\certificado google data analytics\MACHINE LEARNING\Python\metodos supervisados\regresión")
data = pd.read_csv("hormigon.csv")

#Hacemos un analisis exploratorio de los datos
#Vemos las primeras datos
data.head()

#Vemos los tipos de datos que tiene

data.info()

# Realizamos un resumen estadistico de las variables.
data.describe()

#Realizamos un analisis visual de la variable objetivo, strength, es decir la fuera o dureza del cemento
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=data,x="strength")
plt.show()


# Comprobamos que no hay valores perdidos.
data.isnull().sum()



## Una vez que comprobamos que los datos son correctos pasamos a la modelizacion, utilizando tecnicas de machine learning.
# Cuando realizamos un modelo predictivo es necesario conocer la fiabilidad esperada con datos futuros.
# Para ello podemos hacer una particion y utilizar unos datos para entrenar el modelo (train) y otros para comprobar la fiabilidad (test).

# Modelo de regresion #

#Primero realizamos una particion para entrenar el modelo y luego testear 
X=data.drop("strength",axis=1)
y=data["strength"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#importamos la libreria de regresion lineal
from sklearn.linear_model import LinearRegression

#Creamos el modelo
reg = LinearRegression()

#Entrenamos el modelo con los datos de entrenamiento o train
reg.fit(X_train,y_train)
#Evaluamos el modelo con el coeficiente de determinacion R CUADRADO, que cuantifica la varianza de la variable objetivo explicada por las variables explicativas
reg.score(X_train,y_train)

#realizamos la prediccion con los datos de test
y_pred=reg.predict(X_test)
reg.score(X_test,y_test)

#Vemos que el resultado es peor con los datos de test , por lo tanto, podriamos decir que el modelo esta sobreajustado
#El sobreajuste en Reg Lineal sugiere que el modelo se ha ajustado demasiado a los detalles del conjunto de entrenamiento

#Vamos a comprobar si es cierto

reg.get_params(deep=True)
#Esto nos ayuda a saber los coeficientes de las variables explicativas, esos coeficientes son los pesos asignados a cada variable.
reg.coef_
#Esta propiedad nos ayuda a saber si el modelo esta sobreajustado o hay multicolinealidad o hay problemas de singularidad
"""Overfitting: En algunos contextos, un rango cercano al número de características puede indicar overfitting. Un rango elevado podría sugerir que el modelo está utilizando demasiadas características y podría estar sobreajustando los datos de entrenamiento"""
reg.rank_ #El resultado es 8 igual que las variables explicativas>>>Overfitting
#por ultimo vamos a utilizar la siguiente propiedad para saber cuantas variables explicativas estan involucradas en el modelo
reg.n_features_in_ # Resultado 8 

#A continuacion, vamos a proceder con la tecnica de validacion cruazada
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)
cv_results_mean=cv_results.mean()
cv_results_mean

#Parece que no hemos avanzado con este metodo

#Vamos a proceder con otro metodo nuevo 
  # Arbol de decision #

# En el anterior ejemplo hemos realizado una particion de los datos en "orden".
# Otra opcion es realizar una particion aleatoria.
# En este caso tendremos train y test separados aleatoriamente.


# El primer paso es eliminar todo lo creado en el modelo anterior para evitar problemas.
%reset

import os
import pandas as pd
os.chdir(r"C:\Users\PCUser\Desktop\certificado google data analytics\MACHINE LEARNING\Python\metodos supervisados\regresión")
data = pd.read_csv("hormigon.csv")

from sklearn.model_selection import train_test_split
#Primero realizamos una particion para entrenar el modelo y luego testear 
X=data.drop("strength",axis=1)
y=data["strength"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeRegressor

#Creamos el modelo
arbol = DecisionTreeRegressor(criterion='squared_error', 
                              max_depth=8, 
                              max_features=None, 
                              max_leaf_nodes=None,
                              min_impurity_decrease=0.0,
                              min_samples_leaf=10, 
                              min_samples_split=2, 
                              min_weight_fraction_leaf=0.0, 
                              random_state=None, 
                              splitter='best')

arbol.fit(X_train, y_train)
#Evaluamos el modelo con los datos de entrenamiento
arbol.score(X_train,y_train)

#Ahora probamos el modelo con los datos de test

y_pred = arbol.predict(X_test)
arbol.score(X_test,y_test)

#Computamos test MSE
"""el MSE mide la magnitud promedio de los errores cuadrados entre las predicciones del modelo y los valores reales en el conjunto de prueba."""
from sklearn.metrics import mean_squared_error as MSE
mse_dt= MSE(y_test,y_pred)
#computamos el test RMSE
rmse_dt=mse_dt**(1/2)
print(rmse_dt)

y_mean=y.mean()
print(y_mean)

# Random Forest #
#Los Random Forest tienen una mayor resistencia al sobreajuste en comparacion a un solo arbol de decision
#Mejoran la precision y generalizacion del modelo
#Combian multiples arboles de decision de forma aleatoria,usan una tecnica llamada Bootstraping que entrena cada arbol en una muestra aleatoria con reemplazo
#Reducen la varianza y mejora la capacidad de generalizacion del modelo


#Tambien usaremos la metodologia de Cross-Validation


# Lo primero es borrar todo lo existente para evitar problemas.

%reset

import os
import pandas as pd
os.chdir(r"C:\Users\PCUser\Desktop\certificado google data analytics\MACHINE LEARNING\Python\metodos supervisados\regresión")
data = pd.read_csv("hormigon.csv")

from sklearn.model_selection import train_test_split
#Primero realizamos una particion para entrenar el modelo y luego testear 
X=data.drop("strength",axis=1)
y=data["strength"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#Creamos el modelo de ramdom forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
rf=RandomForestRegressor(n_estimators=400,min_samples_leaf=0.12,random_state=3)
#Acoplamos los datos al training set
rf.fit(X_train,y_train)
#predecimos 
y_pred = rf.predict(X_test)
score=rf.score(X_test,y_test)


#Evaluamos con el test MSE
rmse_test = MSE(y_test,y_pred)**(1/2)
#
print('El test RMSE del rf:{:.2f}'.format(rmse_test))
from sklearn.model_selection import cross_val_score
#Si aplicamos la validacion cruzada 
resultados_cv=cross_val_score(rf, X,y,cv=5)
print(resultados_cv.mean())

# El ultimo paso es determinar la importancia de cada una de las variables
import matplotlib.pyplot as plt
importancias=pd.Series(rf.feature_importances_,index=X.columns)
orden_importancias=importancias.sort_values()

orden_importancias.plot(kind="barh",color="lightgreen")
plt.show()

#Se puede observar que la variable que mas peso tiene es la edad seguida por el cement y water

# XGBoost #

# En este caso tambien utilizamos validacion cruzada.

# Lo primero es borrar todo lo existente para evitar problemas.

%reset
y
# En este caso tambien utilizamos validacion cruzada.


# A continuacion volvemos a cargar los datos.


import os
import pandas as pd
os.chdir(r"C:\Users\PCUser\Desktop\certificado google data analytics\MACHINE LEARNING\Python\metodos supervisados\regresión")
data = pd.read_csv("hormigon.csv")

# Separamos la variable dependiente ("y") de las explicativas ("X").

from sklearn.model_selection import train_test_split
#Primero realizamos una particion para entrenar el modelo y luego testear 
X=data.drop("strength",axis=1)
y=data["strength"]

import xgboost as xgb
from sklearn.model_selection import cross_val_score

xgb_model = xgb.XGBRegressor(base_score=1, colsample_bylevel=0.8, colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=1, n_estimators=200, objective='reg:linear', reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=True, subsample=1)
xgb_model

modeloxgb = xgb_model.fit(X,y)
modeloxgb

xgb_model.score(X,y)
scores = cross_val_score(xgb_model, X, y, cv=5)
scores.mean()

from xgboost import plot_importance


#importancias=pd.DataFrame(modeloxgb.feature_importances_)
#importancias.index=(X.columns)


#import pandas as pd
#importacia=pd.concat(X.columns,importancias)

# El ultimo paso es determinar la importancia de cada una de las variables
import matplotlib.pyplot as plt
importancias=pd.Series(xgb_model.feature_importances_,index=X.columns)
orden_importancias=importancias.sort_values()

orden_importancias.plot(kind="barh",color="lightgreen")
plt.show()
