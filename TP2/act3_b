import numpy as np
import matplotlib.pyplot as plt

#Valores iniciales

r = 50
q = 30
a = 0.5
b = 0.5
K = 50

n0 = 30
p0 = 10

n = np.linspace(0, 200, 20)
p = np.linspace(0, 200, 20)

t = np.linspace(0, 150, 20)

plt.figure()

# Isoclinas
iso1 = np.array([q/b] * len(n))
iso2 = np.array([r/a] * len(n))

# plt.subplot(1, 2, 1)
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
plt.show()
