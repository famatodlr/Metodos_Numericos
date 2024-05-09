import numpy as np
import matplotlib.pyplot as plt

# Valores iniciales
N0 = 10
N = np.linspace(0, 1000, 100)
r = 0.1
K = 1000
t = np.linspace(0, 100, 1000)


# Pendiente exponencial
def pendiente_exp(t, n):
    return r * n

# Pendiente logística
def pendiente_log(t, n):
    return r * n * (1 - n/K)

# Gráfica
plt.plot(N, pendiente_exp(0, N), label='Exponencial')
plt.plot(N, pendiente_log(0, N), label='Logística')
plt.xlabel('Población')
plt.ylabel('Tasa de crecimiento')
plt.ylim(0, 100)
plt.legend()
plt.plot([500, 500], [0, 25], 'k--')
plt.plot([0, 500], [25, 25], 'k--')
plt.plot([500], [25], 'ro')
plt.scatter([500], [25], color='red')
plt.show()

def f_exp(t, n0, r):
    return n0 * np.exp((r * t))

def f_log(t, n0, K, r):
    return K / (1 + (K/n0 - 1) * np.exp(-r*t))

#Grafico de f_exp
y_exp = f_exp(t, N0, r)
plt.plot(t, y_exp)
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Modelo Exponencial')

plt.annotate(f'N0 = {N0}\nr = {r}', xy=(0, 1), xycoords='axes fraction', fontsize=12,
             xytext=(10, -10), textcoords='offset points',
             ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

plt.show()

#Grafico de f_log
y_log = f_log(t, N0, K, r)
plt.plot(t, y_log)
plt.xlabel('Tiempo')
plt.plot([45.8, 45.8], [0, 500], 'k--')
plt.plot([0, 45.8], [500, 500], 'k--')
plt.plot([45.8], [500], 'ro')
plt.ylabel('Poblacion')
plt.title('Modelo Logistico')

plt.annotate(f'N0 = {N0}\nK = {K}\nr = {r}', xy=(0, 1), xycoords='axes fraction', fontsize=12,
             xytext=(10, -10), textcoords='offset points',
             ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

plt.show()

def runge_kutta(f, x0, y0, h, n):
    x = x0
    y = y0
    for _ in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        x = x + h
    return y

def euler(f, x0, y0, h, n):
    x = x0
    y = y0
    for _ in range(n):
        y = y + h * f(x, y)
        x = x + h
    return y

def error_absoluto(y, y_hat):
    return abs(y - y_hat)


# Error en Runge-Kutta y Euler funcion exponencial
h = 0.1
n = 100
t = np.linspace(0, 100, n)

y = f_exp(t, N0, r)
y_rk = np.array([runge_kutta(pendiente_exp, 0, N0, h, i) for i in range(n)])
y_euler = np.array([euler(pendiente_exp, 0, N0, h, i) for i in range(n)])

error_rk = error_absoluto(y, y_rk)
error_euler = error_absoluto(y, y_euler)

plt.plot(t, error_rk, label='Runge-Kutta')
plt.plot(t, error_euler, label='Euler')
plt.xlabel('Tiempo')
plt.ylabel('Error')
plt.legend()
plt.title('Error en Runge-Kutta y Euler (Modelo Exponencial)')
plt.show()

# Error en Runge-Kutta y Euler funcion logistica
y = f_log(t, N0, K, r)
y_rk = np.array([runge_kutta(pendiente_log, 0, N0, h, i) for i in range(n)])
y_euler = np.array([euler(pendiente_log, 0, N0, h, i) for i in range(n)])

error_rk = error_absoluto(y, y_rk)
error_euler = error_absoluto(y, y_euler)

plt.plot(t, error_rk, label='Runge-Kutta')
plt.plot(t, error_euler, label='Euler')
plt.xlabel('Tiempo')
plt.ylabel('Error')
plt.legend()
plt.title('Error en Runge-Kutta y Euler (Modelo Logistico)')
plt.show()


