#quiero hacer runge kutta para un sistema de odes de orden 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

''' Consideremos que en el sistema ahora hay dos especies que compiten por los mismos recursos. La ecua-
ción logística ya considera la existencia de competencia intraespecífica, debido a que los recursos se trans-
forman en un limitante a medida que la población se incrementa. Sin embargo, la ecuación logística 4 no es
suficiente si tenemos dos especies en el sistema. Es necesario modelar cómo las dos especies se interrelacio-
nan estableciendo una competencia entre ellas por los recursos (competencia interespecífica). La forma más
sencilla de modelar la competencia interespecífica es mediante un término que reduzca la capacidad de carga
y sea proporcional a la cantidad de individuos de cada especie. De esta forma se definen las ecuaciones de
competencia de Lotka-Volterra'''

#Condiciones iniciales
n1_0 = 100
n2_0 = 100
K1 = 50
K2 = 25
r1 = 0.1
r2 = 0.1
a1 = 0
a2 = 0

def f_n1_prima(n1, n2, r1, K1, a1):
    return r1 * n1 * ((K1 - n1 - a1 * n2) / K1)

def f_n2_prima(n1, n2, r2, K2, a2):
    return r2 * n2 * ((K2 - n2 - a2 * n1) / K2)

def runge_kutta_2(n1, n2, r1, K1, a1, r2, K2, a2, h, t):
    n1 = n1
    n2 = n2
    for _ in range(t):
        k1 = h * f_n1_prima(n1, n2, r1, K1, a1)
        l1 = h * f_n2_prima(n1, n2, r2, K2, a2)
        
        k2 = h * f_n1_prima(n1 + 0.5 * k1, n2 + 0.5 * l1, r1, K1, a1)
        l2 = h * f_n2_prima(n1 + 0.5 * k1, n2 + 0.5 * l1, r2, K2, a2)

        k3 = h * f_n1_prima(n1 + 0.5 * k2, n2 + 0.5 * l2, r1, K1, a1)
        l3 = h * f_n2_prima(n1 + 0.5 * k2, n2 + 0.5 * l2, r2, K2, a2)

        k4 = h * f_n1_prima(n1 + k3, n2 + l3, r1, K1, a1)
        l4 = h * f_n2_prima(n1 + k3, n2 + l3, r2, K2, a2)

        n1 = n1 + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        n2 = n2 + (1/6) * (l1 + 2*l2 + 2*l3 + l4)
    return n1, n2

#graficarunge_kutta_2
t = np.linspace(0, 200, 1000)
n1 = []
n2 = []
for i in t:
    n1_, n2_ = runge_kutta_2(n1_0, n2_0, r1, K1, a1, r2, K2, a2, 0.1, int(i))
    n1.append(n1_)
    n2.append(n2_)
plt.plot(t, n1, label='Especie 1')
plt.plot(t, n2, label='Especie 2')
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Modelo de Competencia de Lotka-Volterra')
plt.legend()
#label de las condiciones iniciales
plt.annotate(f'Condiciones iniciales: n1 = {n1_0}, n2 = {n2_0}, K1 = {K1}, K2 = {K2}, r1 = {r1}, r2 = {r2}, a1 = {a1}, a2 = {a2}',
             
              xy=(0, 1), xycoords='axes fraction', fontsize=12,
             xytext=(10, -10), textcoords='offset points',
             ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

plt.show()


#diagrama de fases
