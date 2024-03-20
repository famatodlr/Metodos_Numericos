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

t = np.arange(len(df))
t_gt = np.arange(len(df_gt))
x1 = df['x1'].values
x2 = df['x2'].values
x1_gt = df_gt['x1'].values
x2_gt = df_gt['x2'].values

# 2. Interpolar los valores de x1 y x2 por separado
# Interpolación de x1
lagrange_poly_x1 = lagrange(t, x1)

# Interpolación de x2
lagrange_poly_x2 = lagrange(t, x2)

# 3. Comparar con los valores de ground truth
# Interpolación de x1
lagrange_poly_x1_gt = lagrange(t_gt, x1_gt)

# Interpolación de x2
lagrange_poly_x2_gt = lagrange(t_gt, x2_gt)

# 4. Graficar
plt.figure(figsize=(12, 6))

# Comparación entre x1 y x2 interpolados y ground truth
plt.subplot(1, 2, 1)
plt.plot(t, x1, label='x1', color='blue')
plt.plot(t, lagrange_poly_x1(t), label='Interpolación de x1', linestyle='--', color='red')
plt.plot(t_gt, x1_gt, label='x1 ground truth', linestyle='-.', color='green')
plt.plot(t_gt, lagrange_poly_x1_gt(t_gt), label='Interpolación de x1 ground truth', linestyle=':', color='purple')
plt.title('Comparación entre x1 y su interpolación')
plt.xlabel('t')
plt.ylabel('x1')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, x2, label='x2', color='blue')
plt.plot(t, lagrange_poly_x2(t), label='Interpolación de x2', linestyle='--', color='red')
plt.plot(t_gt, x2_gt, label='x2 ground truth', linestyle='-.', color='green')
plt.plot(t_gt, lagrange_poly_x2_gt(t_gt), label='Interpolación de x2 ground truth', linestyle=':', color='purple')
plt.title('Comparación entre x2 y su interpolación')
plt.xlabel('t')
plt.ylabel('x2')
plt.legend()
plt.grid(True)

plt.show()
