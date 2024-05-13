import numpy as np
import matplotlib.pyplot as plt

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