'''Se ha registrado la posición de un vehículo autónomo que circula en un parque industrial mediante GPS
en distintos instantes de tiempo (ver archivo mediciones.csv). La posición del vehículo se registra en dos
dimensiones en coordenadas x1(ti), x2(ti).
• Utilizando los datos provistos (mediciones.csv) recupere la trayectoria del vehículo interpolando las
posiciones provistas y compárelas con la trayectoria real (ground truth provisto en groundtruth.csv).'''

# dado mnyo_mediciones.csv, se observa que el archivo tiene una sola columna con los valores de x1 y x2 separados por un espacio.
# se puede leer el archivo con pandas y luego separar los valores de x1 y x2 en dos columnas distintas.
# luego se puede interpolar los valores de x1 y x2 por separado y compararlos con los valores de ground truth.
# aclaracion: no tiene headers x1 y x2, por lo que se debe especificar que no tiene headers al leer el archivo.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline

# 1. Leer los datos
# Leer mediciones.csv
df = pd.read_csv('mnyo_mediciones.csv', header=None, names=['x'])
# Separar los valores de x1 y x2 en dos columnas distintas
df = df['x'].str.split(' ', expand=True)
df.columns = ['x1', 'x2']
# Convertir a float
df = df.astype(float)

# Leer groundtruth.csv
df_gt = pd.read_csv('mnyo_ground_truth.csv', header=None, names=['x'])
# Separar los valores de x1 y x2 en dos columnas distintas
df_gt = df_gt['x'].str.split(' ', expand=True)
df_gt.columns = ['x1', 'x2']
# Convertir a float
df_gt = df_gt.astype(float)

print(df_gt)

# 3. Interpolación
# Interpolación de Lagrange
t = np.arange(len(df))
t_gt = np.arange(len(df_gt))
x1 = df['x1'].values
x2 = df['x2'].values
x1_gt = df_gt['x1'].values
x2_gt = df_gt['x2'].values

