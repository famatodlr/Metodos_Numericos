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

# calcular el error absoluto
error_x = np.abs(x_interpolado - x_gt)
error_y = np.abs(y_interpolado - y_gt)

# Interpolar las posiciones con el metodo de lagrange
interp_x_lagrange = lagrange(tiempo, x)
interp_y_lagrange = lagrange(tiempo, y)

# Interpolamos las posiciones en el conjunto de tiempo continuo
x_interpolado_lagrange = interp_x_lagrange(tiempo_continuo)
y_interpolado_lagrange = interp_y_lagrange(tiempo_continuo)

# calcular el error absoluto
error_x_lagrange = np.abs(x_interpolado_lagrange - x_gt)
error_y_lagrange = np.abs(y_interpolado_lagrange - y_gt)

error_lagrange = np.sqrt(error_x_lagrange**2 + error_y_lagrange**2)
error_spline = np.sqrt(error_x**2 + error_y**2)

# Comparar trayectorias
def grafico1():

    plt.figure(figsize=(12, 8))  # Ajustar el tamaño de la figura

    # Plot de las funciones y puntos de interpolación
    plt.plot(x_gt, y_gt, label='Ground Truth', color='blue')
    plt.plot(x_interpolado, y_interpolado, label='Interpolación con Splines Cúbicos', linestyle='--', color='red')
    plt.plot(x_interpolado_lagrange, y_interpolado_lagrange, label='Interpolación de Lagrange', linestyle='-.', color='orange')
    plt.scatter(x, y, label='Puntos de interpolación', color='green', alpha=0.7, s=50)

    # Etiquetas de los ejes y título
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Comparación de Trayectorias', fontsize=16)

    # Leyenda
    plt.legend(fontsize=12)

    # Cuadrícula
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ajustes adicionales
    plt.gca().spines['top'].set_linewidth(0.5)  # Añadir un borde alrededor del área de trazado
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)

    plt.show()

def grafico2():
        
    plt.figure(figsize=(12, 8))  # Ajustar el tamaño de la figura

    # Gráfico de error absoluto de Lagrange y Spline
    plt.plot(tiempo_continuo, error_lagrange, label='Error absoluto de Lagrange', color='blue')
    plt.plot(tiempo_continuo, error_spline, label='Error absoluto de Spline', color='red')

    # Etiquetas de los ejes y título
    plt.xlabel('Tiempo', fontsize=14)
    plt.ylabel('Error absoluto', fontsize=14)
    plt.title('Error absoluto en la Interpolación con Lagrange y Spline', fontsize=16)

    # Escala logarítmica simétrica en el eje y
    plt.yscale("symlog")
    #hacer que el eje x sea
    plt.xscale("linear")

    # Leyenda
    plt.legend(fontsize=12)

    # Cuadrícula
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ajustes adicionales
    plt.gca().spines['top'].set_linewidth(0.5)  # Añadir un borde alrededor del área de trazado
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)

    plt.show()


grafico2()