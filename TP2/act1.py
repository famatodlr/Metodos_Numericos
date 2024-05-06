import numpy as np
import matplotlib.pyplot as plt

# #Condiciones iniciales
n0 = float(100)
K = float(1000)
r = 0.1

# n0 = float(input('Ingrese la poblacion inicial: '))
# K = float(input('Ingrese la capacidad de carga: '))
# r = float(input('Ingrese la tasa de crecimiento: '))

def f_exp(t, n0, r):
    return n0 * np.exp((r * t))

def f_log(t, n0, K, r):
    return K / (1 + (K/n0 - 1) * np.exp(-r*t))

#Grafico de f_exp
t = np.linspace(0, 100, 1000)
y = f_exp(t, n0, r)
plt.plot(t, y)
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Modelo Exponencial')
#agregar un cuadro con las condiciones iniciales, ubicado en la esquina superior izquierda
plt.annotate(f'n0 = {n0}\nr = {r}', xy=(0, 1), xycoords='axes fraction', fontsize=12,
             xytext=(10, -10), textcoords='offset points',
             ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
plt.show()

#Grafico de f_log
t = np.linspace(0, 100, 1000)
y = f_log(t, n0, K, r)
plt.plot(t, y)
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Modelo Logistico')
#agregar un cuadro con las condiciones iniciales
plt.annotate(f'n0 = {n0}\nK = {K}\nr = {r}', xy=(0, 1), xycoords='axes fraction', fontsize=12,
             xytext=(10, -10), textcoords='offset points',
             ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
plt.show()

