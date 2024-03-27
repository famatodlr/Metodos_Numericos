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
    

def newton_raphson(P0, f = funcion, jac = jac, tol=1e-6, max_iter=100):
    P = P0
    for _ in range(max_iter):
        #calcular la inversa del jacobiano
        J_inv = np.linalg.inv(jac(P[0], P[1]))
        F = f(P[0], P[1])
        P = P - J_inv @ F
        if np.linalg.norm(F) < tol:
            print(f'Iteraciones: {_}')
            return P
        
    return P

P0 = np.array([0, 0])  # Aproximación inicial
punto_cruce = newton_raphson(P0)

r_x = interpA_x(punto_cruce[0])
r_y = interpA_y(punto_cruce[0])

#calculo del error en la interseccion
error = np.linalg.norm(funcion(punto_cruce[0], punto_cruce[1]))
print(f'Error en la intersección: {error}')
print(f'Punto de cruce: ({punto_cruce[0]}, {punto_cruce[1]})')

def grafico1():
    plt.figure(figsize=(12, 8))  # Ajustar el tamaño del gráfico
    plt.scatter(r_x, r_y,  color='green', s=100, marker='o')  # Aumentar el tamaño del marcador y especificar el tipo de marcador
    plt.plot(xA_interpolado, yA_interpolado, label='Trayectoria A', color='blue')
    plt.plot(xB_graficar, yB_graficar, label='Trayectoria B', color='red')
    plt.scatter(xA, yA, color='blue', alpha=0.5)
    plt.scatter(xB, yB, color='red', alpha=0.5)
    plt.scatter(r_x, r_y, color='green', s=100, marker='o')
    plt.xlabel('X1', fontsize=14)  # Aumentar el tamaño de la fuente de la etiqueta X
    plt.ylabel('X2', fontsize=14)  # Aumentar el tamaño de la fuente de la etiqueta Y
    plt.title('Intersección de Trayectorias A y B', fontsize=16)  # Aumentar el tamaño de la fuente del título
    plt.legend(fontsize=12)  # Aumentar el tamaño de la fuente de la leyenda
    plt.grid(True, linestyle='--', alpha=0.7)  # Mejorar el estilo de la cuadrícula
    plt.gca().spines['top'].set_linewidth(0.5)  # Añadir un borde alrededor del área de trazado
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    #agrega a la escala de los ejes el punto exacto de la interseccion
    plt.xticks(list(plt.xticks()[0]) + [r_x])
    plt.yticks(list(plt.yticks()[0]) + [r_y])
    plt.gca().set_facecolor('#f0f0f0')  # Añadir un fondo más agradable

    plt.show()


# grafico1()

