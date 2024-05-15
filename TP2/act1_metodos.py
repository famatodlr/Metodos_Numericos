import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Parámetros
N0 =1  # Condición inicial
r = 0.5  # Tasa de crecimiento
K = 40 # Capacidad de carga
t = np.linspace(0, 200, 75) # Tiempo
h = t[1] - t[0]  #h = 2.6666666666666665
n = len(t)  # Número de puntos

def euler(f, t, N0):
    N = np.zeros(n)
    N[0] = N0
    for i in range(1, n):
        N[i] = N[i - 1] + h * f(N[i - 1], t[i - 1])
    return N

def rk4(f, t, N0):
    N = np.zeros(n)
    N[0] = N0
    for i in range(1, n):
        k1 = h * f(N[i - 1], t[i - 1])
        k2 = h * f(N[i - 1] + k1 / 2, t[i - 1] + h / 2)
        k3 = h * f(N[i - 1] + k2 / 2, t[i - 1] + h / 2)
        k4 = h * f(N[i - 1] + k3, t[i - 1] + h)
        N[i] = N[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return N

def sol_expl(N0, t , r=r,):
    """Solución exacta del modelo exponencial"""
    return N0 * np.exp(r * t)

def dif_exp(N, t , r=r,):
    """Ecuación diferencial del modelo exponencial"""
    return r * N

def sol_log(N0, t,  r=r,  K=K):
    """Solución exacta del modelo logístico"""
    return (N0*K*np.exp(r*t))/((K-N0) + N0*np.exp(r*t))

def dif_log(N, t,  r=r,):
    """Ecuación diferencial del modelo logístico"""
    return r * N * ((K - N) / K)




# Soluciones exactas
N_exp = sol_expl(N0,  t, r)
N_log = sol_log(N0,  t, r, K)

N_euler_exp = euler(dif_exp, t, N0)
N_rk_exp = rk4(dif_exp, t, N0)
N_euler_log = euler(dif_log, t, N0)
N_rk_log = rk4(dif_log, t, N0)

#errores de los metodos
error_euler_exp = np.abs(N_exp - N_euler_exp)
error_rk_exp = np.abs(N_exp - N_rk_exp)
error_euler_log = np.abs(N_log - N_euler_log)
error_rk_log = np.abs(N_log - N_rk_log)

# Gráficos
def grafico_exp():
    plt.plot(t, N_exp, label='Exacta')
    plt.plot(t, N_euler_exp, label='Euler')
    plt.plot(t, N_rk_exp, label='Runge-Kutta')
    plt.ylim(0, 100)

    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Modelo exponencial')
    plt.legend()
    plt.show()

    plt.plot(t, error_euler_exp, label='Error Euler')
    plt.plot(t, error_rk_exp, label='Error Runge-Kutta')
    plt.xlabel('Tiempo')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title('Error modelo exponencial')
    plt.legend()
    plt.show()

def grafico_log():
    plt.plot(t, N_log, label='Exacta')
    plt.plot(t, N_euler_log, label='Euler')
    plt.plot(t, N_rk_log, label='Runge-Kutta')

    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Modelo logístico')
    plt.ylim(0, 50)
    plt.xlim(0, 50)
    plt.legend()
    plt.show()

    plt.plot(t, error_euler_log, label='Error Euler')
    plt.plot(t, error_rk_log, label='Error Runge-Kutta')
    plt.xlabel('Tiempo')
    plt.ylabel('Error')
    plt.yscale('symlog', linthresh=0.001)
    plt.title('Error modelo logístico')
    plt.legend()
    plt.show()

grafico_log()

