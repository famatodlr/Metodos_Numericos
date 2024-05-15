import numpy as np
import matplotlib.pyplot as plt
#condicones iniciales:
N0 = 10
r = 0.05
K = 500

def solucion_exponencial(t):
    return N0 * np.exp(r * t)

def solucion_logistica(t):
    return K * N0 * np.exp(r * t) / ((K - N0) + N0 * np.exp(r * t))

#dame el metodo de euler y de runge kutta para aproximar la solucion de la ecuacion exponencial, usando la solucion exponencial
def euler_exponencial(t0, tf, h):
    t = np.arange(t0, tf, h)
    N = [N0]
    for i in range(1, len(t)):
        N.append(N[i - 1] + h * solucion_exponencial(t[i - 1]))
    return t, N

def runge_kutta_exponencial(t0, tf, h):
    t = np.arange(t0, tf, h)
    N = [N0]
    for i in range(1, len(t)):
        k1 = h * solucion_exponencial(t[i - 1])
        k2 = h * solucion_exponencial(t[i - 1] + h/2)
        k3 = h * solucion_exponencial(t[i - 1] + h/2)
        k4 = h * solucion_exponencial(t[i - 1] + h)
        N.append(N[i - 1] + (k1 + 2*k2 + 2*k3 + k4)/6)
    return t, N

#dame el metodo de euler y de runge kutta para aproximar la solucion de la ecuacion logistica, usando la solucion logistica
def euler_logistica(t0, tf, h):
    t = np.arange(t0, tf, h)
    N = [N0]
    for i in range(1, len(t)):
        N.append(N[i - 1] + h * solucion_logistica(t[i - 1]))
    return t, N

def runge_kutta_logistica(t0, tf, h):
    t = np.arange(t0, tf, h)
    N = [N0]
    for i in range(1, len(t)):
        k1 = h * solucion_logistica(t[i - 1])
        k2 = h * solucion_logistica(t[i - 1] + h/2)
        k3 = h * solucion_logistica(t[i - 1] + h/2)
        k4 = h * solucion_logistica(t[i - 1] + h)
        N.append(N[i - 1] + (k1 + 2*k2 + 2*k3 + k4)/6)
    return t, N

#grafica la solucion de la ecuacion exponencial y sus aproximaciones por euler y runge kutta
def grafica_exponencial():
    t = np.linspace(0, 100, 100)
    N_exp = solucion_exponencial(t)
    t1, N1 = euler_exponencial(0, 100, 1)
    t2, N2 = runge_kutta_exponencial(0, 100, 1)
    plt.plot(t, N_exp, label='Solución exacta')
    plt.plot(t1, N1, label='Euler')
    plt.plot(t2, N2, label='Runge-Kutta')
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Modelo exponencial')
    plt.legend()
    plt.show()

#grafica la solucion de la ecuacion logistica y sus aproximaciones por euler y runge kutta
def grafica_logistica():
    t = np.linspace(0, 100, 100)
    N_log = solucion_logistica(t)
    t1, N1 = euler_logistica(0, 100, 1)
    t2, N2 = runge_kutta_logistica(0, 100, 1)
    plt.plot(t, N_log, label='Solución exacta')
    plt.plot(t1, N1, label='Euler')
    plt.plot(t2, N2, label='Runge-Kutta')
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Modelo logístico')
    plt.legend()
    plt.show()

#grafico de la solucion exponencial y logistica
def grafica():
    t = np.linspace(0, 1000, 500)
    N_exp = solucion_exponencial(t)
    N_log = solucion_logistica(t)
    plt.plot(t, N_exp, label='Exponencial')
    plt.plot(t, N_log, label='Logística')
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Modelos de crecimiento')
    plt.legend()
    plt.show()

grafica()