import numpy as np
import matplotlib.pyplot as plt

#Valores iniciales

r = 2.5
q = 5
a = 0.05
b = 0.1

N0 = 75
P0 = 75

n = np.linspace(-1, 200, 20)
p = np.linspace(-1, 200, 20)

t = np.linspace(0, 200, 100)

plt.figure()

# Isoclinas
iso1 = np.array([q/b] * len(n))
iso2 = np.array([r/a] * len(n))

plt.subplot(1, 2, 1)
plt.plot(iso1, n, label='N = q/b')
plt.plot(n, iso2, label='P = (r/a)')

# Campo vectorial
N,P = np.meshgrid(n, p)

pendiente_N = r * N - a * N * P
pendiente_P = b * N * P - q * P

magnitud = np.sqrt(pendiente_N ** 2 + pendiente_P ** 2)

plt.streamplot(N, P, pendiente_N / magnitud, pendiente_P / magnitud, density=[0.5, 1])

# Grafica
plt.legend()
plt.xlim(0, 200)
plt.ylim(0, 200)
plt.xlabel('Presa')
plt.ylabel('Depredador')
plt.title('Presa vs Depredador')

# Poblaciones en funcion del tiempo
plt.subplot(1, 2, 2)

NP0 = np.array([N0, P0])

def runge_kutta(f, t0, np):
    h = 0.05
    n = 100
    
    t = t0
    y = np

    N1 = [np[0]]
    P1 = [np[1]]

    for _ in range(n - 1):
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t + h

        N1.append(y[0])
        P1.append(y[1])

    return N1, P1

def f(t, y):
    return np.array([r * y[0] - a * y[0] * y[1], b * y[0] * y[1] - q * y[1]])

N1, P1 = runge_kutta(f, 0, NP0)

plt.plot(t, N1, label='Presa')
plt.plot(t, P1, label='Depredador')
plt.legend()
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Poblaciones en función del tiempo')
plt.xlim(0, 200)
plt.ylim(0, 125)
plt.show()
