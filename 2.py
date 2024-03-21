import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
from scipy.interpolate import interp1d


def arreglo_csv(nombre_archivo):
    df = pd.read_csv(nombre_archivo, header=None, names=['x'])
    # Separar los valores de x1 y x2 en dos columnas distintas
    df = df['x'].str.split(' ', expand=True)
    df.columns = ['x1', 'x2']
    # Convertir a float
    df = df.astype(float)
    return df

df = arreglo_csv('mnyo_mediciones.csv')
df_gt = arreglo_csv('mnyo_ground_truth.csv')


x1 = df['x1'].values
x2 = df['x2'].values
x1_gt = df_gt['x1'].values
x2_gt = df_gt['x2'].values

# Asumimos que hay un intervalo de tiempo constante entre cada medición
num_measurements = len(x1)
tiempo = np.linspace(0, num_measurements - 1, num_measurements)  # Creamos el tiempo

# # Paso 2: Interpolar las posiciones
# interp_x1 = interp1d(tiempo, x1, kind='linear')
# interp_x2 = interp1d(tiempo, x2, kind='linear')
interp_x1 = CubicSpline(tiempo, x1)
interp_x2 = CubicSpline(tiempo, x2)


# Creamos un conjunto de tiempo continuo para la interpolación
tiempo_continuo = np.linspace(0, num_measurements - 1, 10*num_measurements)  # Más puntos para una interpolación suave

# Interpolamos las posiciones en el conjunto de tiempo continuo
x1_interpolado = interp_x1(tiempo_continuo)
x2_interpolado = interp_x2(tiempo_continuo)

# Paso 3: calcular el error relativo
error_x1 = np.abs(x1_interpolado - x1_gt) / np.abs(x1_gt)
error_x2 = np.abs(x2_interpolado - x2_gt) / np.abs(x2_gt)

# Graficar el error relativo
plt.figure(figsize=(10, 6))
plt.plot(tiempo_continuo, error_x1, label='Error relativo x1', color='blue')
plt.plot(tiempo_continuo, error_x2, label='Error relativo x2', color='red')
plt.xlabel('Tiempo')
plt.ylabel('Error relativo')
plt.title('Error relativo en la interpolación')
plt.legend()
plt.grid(True)
plt.show()


# Paso 4: Comparar trayectorias
plt.figure(figsize=(10, 6))
plt.plot(x1_gt, x2_gt, label='Ground Truth', color='blue')
plt.plot(x1_interpolado, x2_interpolado, label='Interpolated', linestyle='--', color='red')
plt.scatter(x1, x2, label='Measurements', color='green', alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Comparación de Trayectorias')
plt.legend()
plt.grid(True)
plt.show()







































# # 2. Interpolar los valores de x1 y x2 por separado
# # Interpolación de x1
# lagrange_poly_x1 = lagrange(t, x1)
# # Interpolación de x2
# lagrange_poly_x2 = lagrange(t, x2)
# #interpolacion de los dos interpoladores 
# lagrange_poly_x = lagrange(lagrange_poly_x1, lagrange_poly_x2)





# # 4. Graficar
# plt.figure(figsize=(12, 6))

# # Comparación entre x1 y x2 interpolados y ground truth
# plt.subplot(1, 2, 1)
# plt.plot(t, x1, label='x1', color='blue')
# plt.plot(t, lagrange_poly_x1(t), label='Interpolación de x1', linestyle='--', color='red')
# plt.plot(t_gt, x1_gt, label='x1 ground truth', linestyle='-.', color='green')
# plt.plot(t_gt, lagrange_poly_x1_gt(t_gt), label='Interpolación de x1 ground truth', linestyle=':', color='purple')
# plt.title('Comparación entre x1 y su interpolación')
# plt.xlabel('t')
# plt.ylabel('x1')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(t, x2, label='x2', color='blue')
# plt.plot(t, lagrange_poly_x2(t), label='Interpolación de x2', linestyle='--', color='red')
# plt.plot(t_gt, x2_gt, label='x2 ground truth', linestyle='-.', color='green')
# plt.plot(t_gt, lagrange_poly_x2_gt(t_gt), label='Interpolación de x2 ground truth', linestyle=':', color='purple')
# plt.title('Comparación entre x2 y su interpolación')
# plt.xlabel('t')
# plt.ylabel('x2')
# plt.legend()
# plt.grid(True)

# plt.show()
