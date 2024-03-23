import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
from scipy.interpolate import interp1d


def arreglo_csv(nombre_archivo):
    df = pd.read_csv(nombre_archivo, header=None, names=['x'])
    # Separar los valores de x y y en dos columnas distintas
    df = df['x'].str.split(' ', expand=True)
    df.columns = ['x1', 'x2']
    # Convertir a float
    df = df.astype(float)
    return df

df = arreglo_csv('mnyo_mediciones.csv')
df_gt = arreglo_csv('mnyo_ground_truth.csv')


x = df['x1'].values
y = df['x2'].values
x_gt = df_gt['x1'].values
y_gt = df_gt['x2'].values

# Asumimos que hay un intervalo de tiempo constante entre cada medición
num_measurements = len(x)
tiempo = np.linspace(0, num_measurements - 1, num_measurements)  # Creamos el tiempo

#  Interpolar las posiciones con splines cúbicos
interp_x = CubicSpline(tiempo, x)
interp_y = CubicSpline(tiempo, y)

# Creamos un conjunto de tiempo continuo para la interpolación
tiempo_continuo = np.linspace(0, num_measurements - 1, 10*num_measurements)  # Más puntos para una interpolación suave

# Interpolamos las posiciones en el conjunto de tiempo continuo
x_interpolado = interp_x(tiempo_continuo)
y_interpolado = interp_y(tiempo_continuo)

# calcular el error relativo
error_x = np.abs(x_interpolado - x_gt) / np.abs(x_gt)
error_y = np.abs(y_interpolado - y_gt) / np.abs(y_gt)

# Comparar trayectorias
plt.figure(figsize=(10, 6))
plt.plot(x_gt, y_gt, label='Ground Truth', color='blue')
plt.plot(x_interpolado, y_interpolado, label='Interpolated', linestyle='--', color='red')
plt.scatter(x, y, label='Measurements', color='green', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparación de Trayectorias')
plt.legend()
plt.grid(True)
plt.show()


# Graficar el error relativo
plt.figure(figsize=(10, 6))
plt.plot(tiempo_continuo, error_x, label='Error relativo x', color='blue')
plt.plot(tiempo_continuo, error_y, label='Error relativo y', color='red')
plt.xlabel('Tiempo')
plt.ylabel('Error relativo')
plt.title('Error relativo en la interpolación')
plt.legend()
plt.grid(True)
plt.show()

# Interpolar las posiciones con el metodo de lagrange
interp_x_lagrange = lagrange(tiempo, x)
interp_y_lagrange = lagrange(tiempo, y)

# Interpolamos las posiciones en el conjunto de tiempo continuo
x_interpolado_lagrange = interp_x_lagrange(tiempo_continuo)
y_interpolado_lagrange = interp_y_lagrange(tiempo_continuo)

# calcular el error relativo
error_x_lagrange = np.abs(x_interpolado_lagrange - x_gt) / np.abs(x_gt)
error_y_lagrange = np.abs(y_interpolado_lagrange - y_gt) / np.abs(y_gt)

# Comparar trayectorias
plt.figure(figsize=(10, 6))
plt.plot(x_gt, y_gt, label='Ground Truth', color='blue')
plt.plot(x_interpolado_lagrange, y_interpolado_lagrange, label='Interpolated', linestyle='--', color='red')
plt.scatter(x, y, label='Measurements', color='green', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparación de Trayectorias')
plt.legend()
plt.grid(True)
plt.show()

# Graficar el error relativo
plt.figure(figsize=(10, 6))
plt.plot(tiempo_continuo, error_x_lagrange, label='Error relativo x', color='blue')
plt.plot(tiempo_continuo, error_y_lagrange, label='Error relativo y', color='red')
plt.xlabel('Tiempo')
plt.ylabel('Error relativo')
plt.title('Error relativo en la interpolación')
plt.legend()
plt.grid(True)
plt.show()