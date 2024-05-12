import numpy as np
import matplotlib.pyplot as plt

# # Valores iniciales
# datos_iniciales = {
#     'N0': 25,
#     'r': 0.05,
#     'K': 500
# }

# # Pendiente exponencial
# def pendiente_exp(t, n):
#     return datos_iniciales['r'] * n

# # Pendiente logística
# def pendiente_log(t, n):
#     return datos_iniciales['r'] * n * (1 - n/ datos_iniciales['K'])

# # Gráfica
# def grafico1():
#     N = np.linspace(0, 500, 100)

#     plt.plot(N, pendiente_exp(0, N), label='Exponencial')
#     plt.plot(N, pendiente_log(0, N), label='Logística')
#     plt.xlabel('Población')
#     plt.ylabel('Tasa de crecimiento')
#     plt.title('Pendiente de crecimiento')
#     # plt.ylim(0, 100)
#     plt.legend()
#     plt.plot([500, 500], [0, 25], 'k--')
#     plt.plot([0, 500], [25, 25], 'k--')
#     plt.plot([500], [25], 'ro')
#     plt.scatter([500], [25], color='red')
#     plt.show()

# def f_exp(t, n0, r):
#     return n0 * np.exp((r * t))

# def f_log(t, n0, K, r):
    # https://courses.lumenlearning.com/calculus2/chapter/solving-the-logistic-differential-equation/

#     # return K / (1 + (K/n0 - 1) * np.exp(-r*t))
#     return n0 * K * np.exp(r * t)/ ((K - n0) + n0 * np.exp(-r * t))  

# #Grafico de f_exp
# def grafico2():
#     y_exp = f_exp(t, datos_iniciales['N0'], datos_iniciales['r'])
#     plt.plot(t, y_exp)
#     plt.xlabel('Tiempo')
#     plt.ylabel('Poblacion')
#     plt.title('Modelo Exponencial')

#     plt.annotate(f'N0 = {datos_iniciales["N0"]}\nr = {datos_iniciales["r"]}', xy=(0, 1), xycoords='axes fraction', fontsize=12,
#                 xytext=(10, -10), textcoords='offset points',
#                 ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

#     plt.show()

# #Grafico de f_log
# def grafico3():
#     y_log = f_log(t, datos_iniciales['N0'], datos_iniciales['K'], datos_iniciales['r'])
#     plt.plot(t, y_log)
#     plt.xlabel('Tiempo')
#     plt.plot([45.8, 45.8], [0, 500], 'k--')
#     plt.plot([0, 45.8], [500, 500], 'k--')
#     plt.plot([45.8], [500], 'ro')
#     plt.ylabel('Poblacion')
#     plt.title('Modelo Logistico')

#     plt.annotate(f'N0 = {datos_iniciales["N0"]}\nK = {datos_iniciales["K"]}\nr = {datos_iniciales["r"]}', xy=(0, 1), xycoords='axes fraction', fontsize=12,
#                 xytext=(10, -10), textcoords='offset points',
#                 ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

#     plt.show()

# def runge_kutta(f, x0, y0, h, n):
#     x = x0
#     y = y0
#     for _ in range(n):
#         k1 = h * f(x, y)
#         k2 = h * f(x + h/2, y + k1/2)
#         k3 = h * f(x + h/2, y + k2/2)
#         k4 = h * f(x + h, y + k3)
#         y = y + (k1 + 2*k2 + 2*k3 + k4)/6
#         x = x + h
#     return y

# def euler(f, x0, y0, h, n):
#     x = x0
#     y = y0
#     for _ in range(n):
#         y = y + h * f(x, y)
#         x = x + h
#     return y

# def error_absoluto(y, y_hat):
#     return abs(y - y_hat)


# # Error en Runge-Kutta y Euler funcion exponencial
# h = 0.1
# n = 100
# t = np.linspace(0, 100, n)

# y = f_exp(t, datos_iniciales['N0'], datos_iniciales['r'])
# y_rk = np.array([runge_kutta(pendiente_exp, 0, datos_iniciales['N0'], h, i) for i in range(n)])
# y_euler = np.array([euler(pendiente_exp, 0,datos_iniciales['N0'], h, i) for i in range(n)])

# error_rk = error_absoluto(y, y_rk)
# error_euler = error_absoluto(y, y_euler)

# plt.plot(t, error_rk, label='Runge-Kutta')
# plt.plot(t, error_euler, label='Euler')
# plt.xlabel('Tiempo')
# plt.ylabel('Error')
# plt.yscale('log')
# plt.legend()
# plt.title('Error en Runge-Kutta y Euler (Modelo Exponencial)')
# plt.show()

# # Error en Runge-Kutta y Euler funcion logistica
# y = f_log(t, datos_iniciales['N0'], datos_iniciales['K'], datos_iniciales['r'])
# y_rk = np.array([runge_kutta(pendiente_log, 0, datos_iniciales['N0'], h, i) for i in range(n)])
# y_euler = np.array([euler(pendiente_log, 0, datos_iniciales['N0'], h, i) for i in range(n)])

# error_rk = error_absoluto(y, y_rk)
# error_euler = error_absoluto(y, y_euler)

# plt.plot(t, error_rk, label='Runge-Kutta')
# plt.plot(t, error_euler, label='Euler')
# plt.xlabel('Tiempo')
# plt.ylabel('Error')
# plt.yscale('log')
# plt.legend()
# plt.title('Error en Runge-Kutta y Euler (Modelo Logistico)')
# plt.show()

def variacion_poblacional_exponencial(r, N):
    return r * N

def variacion_poblacional_logistica(N, K, r):
    return r * N * ((K - N) / K)

def solucion_exponencial(t, N0, r):
    return N0 * np.exp(r * t)

def solucion_logistica(t, N0, K, r):
    return K * N0 * np.exp(r * t) / ((K - N0) + N0 * np.exp(r * t))

#N0 y K >= 0 y r cualquier valor en reales
#Caso 1: N0 = K => en el modelo logistico, no hay crecimiento
#Caso 2: N0 < K => en el modelo logistico, hay crecimiento hasta llegar a K con una velocidad r
#Caso 3: N0 > K => en el modelo logistico, hay decrecimiento hasta llegar a K
#Caso 4: N0 = 0 => en el modelo logistico, no hay crecimiento
#Caso 5: K = 0 => en el modelo logistico, no hay crecimiento
#Caso 6: r = 0 => en el modelo logistico, no hay crecimiento
#Caso 7: r < 0 => en el modelo logistico, hay decrecimiento
#Caso 8: r > 0 => en el modelo logistico, hay crecimiento

def caso2():
    t = np.linspace(0, 100, 100)
    N0 = 10
    r = 0.05
    K = 100

    N_exp = solucion_exponencial(t, N0, r)
    N_log = solucion_logistica(t, N0, K, r)

    plt.plot(t, N_exp, label='Exponencial')
    plt.plot(t, N_log, label='Logística')
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Modelos de crecimiento')
    plt.legend()
    plt.show()


def caso3():
    t = np.linspace(0, 100, 100)
    N0 = 350
    r = 0.05
    K = 100

    N_exp = solucion_exponencial(t, N0, r)
    N_log = solucion_logistica(t, N0, K, r)

    # plt.plot(t, N_exp, label='Exponencial')
    plt.plot(t, N_log, label='Logística')
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.ylim(0, 400)
    plt.title('Modelos de crecimiento')
    plt.legend()
    plt.show()

def caso1VN():
    N = np.linspace(0, 100, 100)
    r = 0.05
    K = 100

    N_exp = variacion_poblacional_exponencial(r, N)
    N_log = variacion_poblacional_logistica(N, K, r)

    plt.plot(N, N_exp, label='Exponencial')
    plt.plot(N, N_log, label='Logística')
    plt.xlabel('Población')
    plt.ylabel('Variación poblacional')
    plt.legend()
    plt.show()

def caso2VN():
    N = np.linspace(0, 1000, 100)
    r = 0.05
    K = 1000

    N_exp = variacion_poblacional_exponencial(r, N)
    N_log = variacion_poblacional_logistica(N, K, r)

    plt.plot(N, N_exp, label='Exponencial')
    plt.plot(N, N_log, label='Logística')
    plt.xlabel('Población')
    plt.xlim(0, 1000)\
    #marcar el punto (500, 12.5), y poner lineas punteadas en x y y, y poner una etiqueta en el punto
    plt.plot([500, 500], [0, 12.5], 'k--')
    plt.plot([0, 500], [12.5, 12.5], 'k--')
    plt.plot([500], [12.5], 'ro')
    plt.scatter([500], [12.5], color='red')
    #marcar los valores 12,5 y 500 en el eje x y y

    plt.yticks([12.5], ['12.5'])
    plt.xticks([500], ['500'])
    plt.ylabel('Variación poblacional')
    plt.legend()
    plt.show()


#SEGUNDA PARTE
#Soluciones numericas de ambas ecuaciones diferenciales por metodo de Runge-Kutta de orden 4 y Euler
def runge_kutta(f, t0, y0, h, n):
    t = t0
    y = y0
    for _ in range(n):
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t + h
    return y

def euler(f, t0, y0, h, n):
    t = t0
    y = y0
    for _ in range(n):
        y = y + h * f(t, y)
        t = t + h
    return y

def sol_num_Exp():
    N0 = 25
    r = 0.05
    t0 = 0
    h = 0.1
    n = 100

    t = np.linspace(t0, t0 + h * n, n)
    y = solucion_exponencial(t, N0, r)

    y_rk = np.array([runge_kutta(variacion_poblacional_exponencial, t0, N0, h, i) for i in range(n)])
    y_euler = np.array([euler(variacion_poblacional_exponencial, t0, N0, h, i) for i in range(n)])

    plt.plot(t, y, label='Solución exacta')
    plt.plot(t, y_rk, label='Runge-Kutta')
    plt.plot(t, y_euler, label='Euler')
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Modelo Exponencial')
    plt.legend()
    plt.show()

def sol_num_Log():
    N0 = 25
    K = 500
    r = 0.05
    t0 = 0
    h = 0.1
    n = 100

    t = np.linspace(t0, t0 + h * n, n)
    y = solucion_logistica(t, N0, K, r)

    y_rk = np.array([runge_kutta(variacion_poblacional_logistica, t0, N0, h, i) for i in range(n)])
    y_euler = np.array([euler(variacion_poblacional_logistica, t0, N0, h, i) for i in range(n)])

    plt.plot(t, y, label='Solución exacta')
    plt.plot(t, y_rk, label='Runge-Kutta')
    plt.plot(t, y_euler, label='Euler')
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Modelo Logístico')
    plt.legend()
    plt.show()