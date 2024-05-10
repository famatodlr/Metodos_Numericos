import numpy as np
import matplotlib.pyplot as plt

#Valores iniciales
casos = {
        'Caso 1': [200, 50, 2, 0.5],
        'Caso 2': [100, 100, 2, 0.5],
        'Caso 3': [200, 140, 2.5, 1],
        'Caso 4': [100, 60, 0.2, 0.2]}
        #'Caso X': [K1, K2, a12, a21]        

N1_0 = 30
N2_0 = 10

r1 = 0.6
r2 = 0.6

xlim1 = 200
ylim1 = 140
n = 30

xlim2 = 150
ylim2 = 175

x = np.linspace(0, xlim1, n)
y = np.linspace(0, ylim1, n)

for key, value in casos.items():
    K1, K2, a12, a21 = value
    plt.figure()
    plt.suptitle(key)
    
    # Rectas de las isoclinas

    n1 = K1 - a12 * y
    n2 = K2 - a21 * x

    plt.subplot(1, 2, 1)
    plt.plot(n1, y, label='K1 - a12 * N2 = 0')
    plt.plot(x, n2, label='K2 - a21 * N1 = 0')

    # Campo vectorial
    N1,N2 = np.meshgrid(x, y)

    pendiente_N1 =r1 * N1 * (K1 - N1 - a12 * N2) / K1
    pendiente_N2 = r2 * N2 * (K2 - N2 - a21 * N1) / K2

    magnitud = np.sqrt(pendiente_N1 ** 2 + pendiente_N2 ** 2)

    plt.streamplot(N1, N2, pendiente_N1 / magnitud, pendiente_N2 / magnitud, density=[0.5, 1])

    # Grafica 
    plt.legend()
    plt.xlim(0, xlim1)
    plt.ylim(0, ylim1)
    plt.xlabel('Tasa de crecimiento población 1')
    plt.ylabel('Tasa de crecimiento población 2')


    # Poblaciones en funcion del tiempo
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

    def f_N1(t, N):
        return r1 * N[0] * (K1 - N[0] - a12 * N[1]) / K1
    
    def f_N2(t, N):
        return r2 * N[1] * (K2 - N[1] - a21 * N[0]) / K2
    
    t = np.linspace(0, 150, 100)
    N = np.zeros((2, len(t)))
    N[:,0] = [N1_0, N2_0]

    for i in range(1, len(t)):
        N[:,i] = runge_kutta(lambda t, N: np.array([f_N1(t, N), f_N2(t, N)]), 0, N[:,i-1], 0.1, 1)

    
    plt.subplot(1, 2, 2)
    plt.plot(t, N[0], label='Poblacioón 1')
    plt.plot(t, N[1], label='Poblacioón 2')
    plt.legend()
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.xlim(0, xlim2)
    plt.ylim(0, ylim2)
    
    plt.show()

