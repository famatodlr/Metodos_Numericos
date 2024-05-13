import numpy as np
import matplotlib.pyplot as plt

#Valores iniciales

r = 75
q = 50
a = 0.5
b = 0.5
K = 50

n0 = 150
p0 = 150

n = np.linspace(0, 200, 20)
p = np.linspace(0, 200, 20)



plt.figure()

# Isoclinas
iso1 = np.array([q/b] * len(n))
iso2 = r/a * (1 - n/K)

plt.subplot(1, 2, 1)
plt.plot(iso1, n, label='N = q/b')
plt.plot(n, iso2, label='P = r/a(1 - N/K)')

# Campo vectorial
N,P = np.meshgrid(n, p)

pendiente_N = r * N * (1 - N / K) - a * N * P
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

NP0 = np.array([n0, p0])

h = 0.005
n = 51
j = int(n // h)

t = np.linspace(0, 150, j)

def runge_kutta(f, np):
    
    t = 0
    y = np

    N1 = []
    P1 = []

    N1.append(y[0])
    P1.append(y[1])

    for _ in range(j - 1):
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
    return np.array([r*y[0]*(1 - y[0]/K) - a*y[0]*y[1], b*y[0]*y[1] - q*y[1]])

N1, P1 = runge_kutta(f, NP0)

plt.plot(t, N1, label='Presa')
plt.plot(t, P1, label='Depredador')
plt.legend()
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Poblaciones en función del tiempo')
plt.xlim(0, 1)
plt.ylim(0, 300)
plt.show()

