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

xA = dfA['x1'].values
yA = dfA['x2'].values

xB = dfB['x1'].values
yB = dfB['x2'].values

# Asumimos que hay un intervalo de tiempo constante entre cada medición
tiempoA = np.linspace(0, len(xA) - 1, len(xA))
tiempoB = np.linspace(0, len(xB) - 1, len(xB))

tiempo_continuoA = np.linspace(0, len(xA) - 1, 10*len(xA)) # Más puntos para una interpolación suave
tiempo_continuoB = np.linspace(0, len(xB) - 1, 10*len(xB))

interpA_x = CubicSpline(tiempoA, xA)
interpA_y = CubicSpline(tiempoA, yA)
xA_interpolado = interpA_x(tiempo_continuoA)
yA_interpolado = interpA_y(tiempo_continuoA)

interpB_x = CubicSpline(tiempoB, xB)
interpB_y = CubicSpline(tiempoB, yB)
xB_interpolado = interpB_x(tiempo_continuoA)
yB_interpolado = interpB_y(tiempo_continuoA)
xB_graficar = interpB_x(tiempo_continuoB)
yB_graficar = interpB_y(tiempo_continuoB)

def jac(t1, t2):
    return np.array([[interpA_x.derivative()(t1), -interpB_x.derivative()(t2)], [interpA_y.derivative()(t1), -interpB_y.derivative()(t2)]])

def funcion(t1, t2):
    return np.array([interpA_x(t1) - interpB_x(t2), interpA_y(t1) - interpB_y(t2)])
    

def newton_raphson(P0, f = funcion, jac = jac, tol=1e-20, max_iter=100):
    P = P0
    iter = 0
    for _ in range(max_iter):
        #calcular la inversa del jacobiano
        J_inv = np.linalg.inv(jac(P[0], P[1]))
        F = f(P[0], P[1])
        P = P - J_inv @ F
        if np.linalg.norm(F) < tol:
            iter = _
            print(f'Iteración {iter}: {P}')
            return P
        
    return P

P0 = np.array([0, 0])  # Aproximación inicial
punto_cruce = newton_raphson(P0)

r_x = interpA_x(punto_cruce[0])
r_y = interpA_y(punto_cruce[0])


# Comparar trayectorias
plt.figure(figsize=(10, 6))
plt.plot(xA_interpolado, yA_interpolado, label='Funcion A', color='blue')
plt.plot(xB_graficar, yB_graficar, label='Funcion B', color='red')
plt.scatter(xA, yA, label='Measurements A', color='blue', alpha=0.5)
plt.scatter(xB, yB, label='Measurements B', color='red', alpha=0.5)
plt.scatter(r_x, r_y, label='Intersection', color='green', s=50)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Comparación de Trayectorias')
plt.legend()
plt.grid(True)
plt.show()