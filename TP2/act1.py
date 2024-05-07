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

#metodo de runge kutta para la ecuacion exponencial

def runge_kutta_exp(n0, r, h, t):
    n = n0
    for i in range(t):
        k1 = r * n
        k2 = r * (n + 0.5 * h * k1)
        k3 = r * (n + 0.5 * h * k2)
        k4 = r * (n + h * k3)
        n = n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return n

#metodo de runge kutta para la ecuacion logistica

def runge_kutta_log(n0, K, r, h, t):
    n = n0
    for i in range(t):
        k1 = r * n * (1 - n/K)
        k2 = r * (n + 0.5 * h * k1) * (1 - (n + 0.5 * h * k1)/K)
        k3 = r * (n + 0.5 * h * k2) * (1 - (n + 0.5 * h * k2)/K)
        k4 = r * (n + h * k3) * (1 - (n + h * k3)/K)
        n = n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return n

#Grafico de runge_kutta_exp
t = np.linspace(0, 100, 1000)
y = [runge_kutta_exp(n0, r, 0.1, i) for i in range(1000)]
plt.plot(t, y)
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Modelo Exponencial Runge Kutta')

plt.show()

#Grafico de runge_kutta_log
t = np.linspace(0, 100, 1000)
y = [runge_kutta_log(n0, K, r, 0.1, i) for i in range(1000)]
plt.plot(t, y)
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Modelo Logistico Runge Kutta')

plt.show()

# #Comparacion de los modelos
t = np.linspace(0, 100, 1000)
y_exp = f_exp(t, n0, r)
y_log = f_log(t, n0, K, r)
y_rk_exp = [runge_kutta_exp(n0, r, 0.1, i) for i in range(1000)]
y_rk_log = [runge_kutta_log(n0, K, r, 0.1, i) for i in range(1000)]

plt.plot(t, y_exp, label='Exponencial')
plt.plot(t, y_log, label='Logistico')
plt.plot(t, y_rk_exp, label='Exponencial Runge Kutta')
plt.plot(t, y_rk_log, label='Logistico Runge Kutta')
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Comparacion de modelos')
plt.legend()
plt.show()

# metodo de euler para la ecuacion exponencial

def euler_exp(n0, r, h, t):
    n = n0
    for i in range(t):
        n = n + h * r * n
    return n

# metodo de euler para la ecuacion logistica

def euler_log(n0, K, r, h, t):
    n = n0
    for i in range(t):
        n = n + h * r * n * (1 - n/K)
    return n

# Grafico de euler_exp
t = np.linspace(0, 100, 1000)
y = [euler_exp(n0, r, 0.1, i) for i in range(1000)]
plt.plot(t, y)
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Modelo Exponencial Euler')

plt.show()

# Grafico de euler_log
t = np.linspace(0, 100, 1000)
y = [euler_log(n0, K, r, 0.1, i) for i in range(1000)]
plt.plot(t, y)
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Modelo Logistico Euler')

plt.show()

# Comparacion de los modelos
t = np.linspace(0, 100, 1000)
y_exp = f_exp(t, n0, r)
y_log = f_log(t, n0, K, r)
y_rk_exp = [runge_kutta_exp(n0, r, 0.1, i) for i in range(1000)]
y_rk_log = [runge_kutta_log(n0, K, r, 0.1, i) for i in range(1000)]
y_euler_exp = [euler_exp(n0, r, 0.1, i) for i in range(1000)]
y_euler_log = [euler_log(n0, K, r, 0.1, i) for i in range(1000)]

plt.plot(t, y_exp, label='Exponencial')
plt.plot(t, y_log, label='Logistico')
plt.plot(t, y_rk_exp, label='Exponencial Runge Kutta')
plt.plot(t, y_rk_log, label='Logistico Runge Kutta')
plt.plot(t, y_euler_exp, label='Exponencial Euler')
plt.plot(t, y_euler_log, label='Logistico Euler')
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Comparacion de modelos')
plt.legend()
plt.show()

