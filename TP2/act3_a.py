import numpy as np
import matplotlib.pyplot as plt

#Valores iniciales

r = 2.5
q = 5
a = 0.05
b = 0.1

n = np.linspace(-50, 200, 20)
p = np.linspace(-50, 200, 20)

t = np.linspace(0, 200, 100)

plt.figure()

# Isoclinas
iso1 = np.array([q/b] * len(n))
iso2 = np.array([r/a] * len(n))

plt.plot(iso1, n, label='N = q/b')
plt.plot(n, iso2, label='P = (r/a)')

# Grafica
plt.legend()
plt.xlim(-10, 200)
plt.ylim(-10, 200)
plt.xlabel('Presa')
plt.ylabel('Depredador')
plt.title('Presa vs Depredador')

# Poblaciones en funcion del tiempo
NP0_2 = np.array([50, 50])
NP0_3 = np.array([25, 25])
NP0_4 = np.array([40, 60])
NP0_5 = np.array([125, 25])
NP0_6 = np.array([75, 75])
NP0_7 = np.array([0, 75])
NP0_8 = np.array([75, 0])

def runge_kutta(f, t0, np):
    h = 0.005
    n = 500
    
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

N2, P2 = runge_kutta(f, 0, NP0_2)
N3, P3 = runge_kutta(f, 0, NP0_3)
N4, P4 = runge_kutta(f, 0, NP0_4)
N5, P5 = runge_kutta(f, 0, NP0_5)
N6, P6 = runge_kutta(f, 0, NP0_6)
N7, P7 = runge_kutta(f, 0, NP0_7)
N8, P8 = runge_kutta(f, 0, NP0_8)

frac = 100

plt.plot(N2, P2, color='black')
plt.arrow(N2[len(N2)//frac], P2[len(P2)//frac], N2[len(N2)//frac + 1] - N2[len(N2)//frac], P2[len(P2)//frac + 1] - P2[len(P2)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

plt.plot(N3, P3, color='black')
plt.arrow(N3[len(N3)//frac], P3[len(P3)//frac], N3[len(N3)//frac + 1] - N3[len(N3)//frac], P3[len(P3)//frac + 1] - P3[len(P3)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

plt.plot(N4, P4, color='black')
plt.arrow(N4[len(N4)//frac], P4[len(P4)//frac], N4[len(N4)//frac + 1] - N4[len(N4)//frac], P4[len(P4)//frac + 1] - P4[len(P4)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

plt.plot(N5, P5, color='black')
plt.arrow(N5[len(N5)//frac], P5[len(P5)//frac], N5[len(N5)//frac + 1] - N5[len(N5)//frac], P5[len(P5)//frac + 1] - P5[len(P5)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

plt.plot(N6, P6, color='black')
plt.arrow(N6[len(N6)//frac], P6[len(P6)//frac], N6[len(N6)//frac + 1] - N6[len(N6)//frac], P6[len(P6)//frac + 1] - P6[len(P6)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

plt.plot(N7, P7, color='black')
plt.arrow(N7[len(N7)//frac], P7[len(P7)//frac], N7[len(N7)//frac + 1] - N7[len(N7)//frac], P7[len(P7)//frac + 1] - P7[len(P7)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

plt.plot(N8, P8, color='black')
plt.arrow(N8[len(N8)//frac], P8[len(P8)//frac], N8[len(N8)//frac + 1] - N8[len(N8)//frac], P8[len(P8)//frac + 1] - P8[len(P8)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

plt.show()

