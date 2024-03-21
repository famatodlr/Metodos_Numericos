import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import newton


def arreglo_csv(nombre_archivo):
    df = pd.read_csv(nombre_archivo, header=None, names=['x'])
    # Separar los valores de x1 y x2 en dos columnas distintas
    df = df['x'].str.split(' ', expand=True)
    df.columns = ['x1', 'x2']
    # Convertir a float
    df = df.astype(float)
    return df


dfA = arreglo_csv('mnyo_mediciones.csv')
dfB = arreglo_csv('mnyo_mediciones2.csv')

x1A = dfA['x1'].values
x2A = dfA['x2'].values
x1B = dfB['x1'].values
x2B = dfB['x2'].values

# Asumimos que hay un intervalo de tiempo constante entre cada medición
num_measurements = len(x1B)
tiempoB = np.linspace(0, len(x1B) - 1, len(x1B))
tiempoA = np.linspace(0, len(x1A) - 1, len(x1A))

tiempo_continuoA = np.linspace(0, len(x1A) - 1, 10*len(x1A)) # Más puntos para una interpolación suave
tiempo_continuoB = np.linspace(0, len(x1B) - 1, 10*len(x1B)) # Más puntos para una interpolación suave

interpA_x1 = CubicSpline(tiempoA, x1A)
interpA_x2 = CubicSpline(tiempoA, x2A)
x1A_interpolado = interpA_x1(tiempo_continuoA)
x2A_interpolado = interpA_x2(tiempo_continuoA)

interpB_x1 = CubicSpline(tiempoB, x1B)
interpB_x2 = CubicSpline(tiempoB, x2B)
x1B_interpolado = interpB_x1(tiempo_continuoB)
x2B_interpolado = interpB_x2(tiempo_continuoB)

# Se define el metodo de newton 
def metodo_newton(f, df, x0, tol, max_iter):
    x = x0
    for i in range(max_iter):
        x = x - f(x) / df(x)
        if abs(f(x)) < tol:
            return x
    return x

# Se define el intervalo de busqueda
x0 = 0
x1 = 10
tol = 1e-6
max_iter = 100

# Se define la resta de los polinomios 1
def resta_pol1(x):
    return interpA_x1(x) - interpB_x1(x)


# Se define la derivada de la resta de los polinomios 1
def derivada_resta_pol1(x):
    return interpA_x1.derivative()(x) - interpB_x1.derivative()(x)


# Se aplica el metodo de newton 1
raiz1 = metodo_newton(resta_pol1, derivada_resta_pol1, x0, tol, max_iter)
r1 = interpA_x1(raiz1)


# Se grafica la raiz 1
plt.figure(figsize=(10, 6))
plt.plot(tiempo_continuoA, resta_pol1(tiempo_continuoA), label='Resta de polinomios', color='blue')
plt.scatter(raiz1, resta_pol1(raiz1), label='Raiz', color='red')
plt.xlabel('Tiempo')
plt.ylabel('Resta de polinomios')
plt.title('Raiz de la resta de polinomios')
plt.legend()
plt.grid(True)
plt.show()


# se define la resta de los polimonios 2
def resta_pol2(x):
    return interpA_x2(x) - interpB_x2(x)

# Se define la derivada de la resta de los polinomios 2
def derivada_resta_pol2(x):
    return interpA_x2.derivative()(x) - interpB_x2.derivative()(x)

# Se aplica el metodo de newton 2
raiz2 = metodo_newton(resta_pol2, derivada_resta_pol2, x0, tol, max_iter)
r2 = interpA_x2(raiz2)


# Paso 4: Comparar trayectorias
plt.figure(figsize=(10, 6))
plt.plot(x1A_interpolado, x2A_interpolado, label='Funcion A', color='blue')
plt.plot(x1B_interpolado, x2B_interpolado, label='Funcion B', linestyle='--', color='red')
plt.scatter(x1A, x2A, label='Measurements A', color='green', alpha=0.5)
plt.scatter(x1B, x2B, label='Measurements B', color='green', alpha=0.5)
plt.scatter(r1, r2, label='Intersection', color='blue', alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Comparación de Trayectorias')
plt.legend()
plt.grid(True)
plt.show()