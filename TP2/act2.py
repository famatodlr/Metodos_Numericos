import numpy as np
import matplotlib.pyplot as plt

#puntos fijos
#caso 1: trivial (0,0)
#caso 2: igualacion de las isoclinas
#caso 3: K1 o k2 = 0
#r = la velocidad a la que se acercan los puntos fijos

#Valores iniciales
casos = {
        'Caso 1': [200, 50, 2, 0.5],
        'Caso 2': [100, 100, 2, 0.5],
        'Caso 3': [200, 100, 5, 0.75],
        'Caso 4': [100, 60, 0.2, 0.2]}
        #'Caso X': [K1, K2, a12, a21]        

r1 = 0.6
r2 = 0.6

xlim1 = 200
ylim1 = 120
n = 100
t = np.linspace(0, 250, 100)

x = np.linspace(0, xlim1, n)
y = np.linspace(0, ylim1, n)

def runge_kutta(f, n0, caso):
    K1, K2, a12, a21 = casos[caso]
    h = 0.001
    n = 150
    
    t = 0

    y = n0

    N1 = []
    P1 = []

    N1.append(y[0])
    P1.append(y[1])

    for _ in range(n - 1):
        k1 = h * f(y, K1, K2, a12, a21)
        k2 = h * f(y + k1/2, K1, K2, a12, a21)
        k3 = h * f(y + k2/2, K1, K2, a12, a21)
        k4 = h * f(y + k3, K1, K2, a12, a21)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t + h

        N1.append(y[0])
        P1.append(y[1])

    return N1, P1

def f(y, k1, k2, a12, a21):
    return np.array([r1 * y[0] * (k1 - y[0] - a12 * y[1]), r2 * y[1] * (k2 - y[1] - a21 * y[0])])

def caso1():
    K1, K2, a12, a21 = casos['Caso 1']

    # Rectas de las isoclinas
    iso1 = K1 - a12 * y       # ordenada al origen --> K1 / a12       #raiz --> K1
    iso2 = K2 - a21 * x       # ordenada al origen --> K2             #raiz --> K2 / a21

    # Poblaciones en funcion del tiempo
    NP0_1 = np.array([25, 25])
    N1_1, N2_1 = runge_kutta(f, NP0_1, 'Caso 1')

    NP0_2 = np.array([50, 50])
    N1_2, N2_2 = runge_kutta(f, NP0_2, 'Caso 1')

    NP0_3 = np.array([100, 100])
    N1_3, N2_3 = runge_kutta(f, NP0_3, 'Caso 1')

    frac = 100
    
    plt.figure()
    plt.plot(iso1, y, label='Iso1')
    plt.plot(x, iso2, label='Iso2')
    plt.plot(N1_1, N2_1, color='black')

    plt.arrow(N1_1[len(N1_1)//frac], N2_1[len(N2_1)//frac], N1_1[len(N1_1)//frac + 1] - N1_1[len(N1_1)//frac], N2_1[len(N2_1)//frac + 1] - N2_1[len(N2_1)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_2, N2_2, color='black')

    plt.arrow(N1_2[len(N1_2)//frac], N2_2[len(N2_2)//frac], N1_2[len(N1_2)//frac + 1] - N1_2[len(N1_2)//frac], N2_2[len(N2_2)//frac + 1] - N2_2[len(N2_2)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_3, N2_3, color='black')

    plt.arrow(N1_3[len(N1_3)//frac], N2_3[len(N2_3)//frac], N1_3[len(N1_3)//frac + 1] - N1_3[len(N1_3)//frac], N2_3[len(N2_3)//frac + 1] - N2_3[len(N2_3)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.xlim(0, xlim1)
    plt.ylim(0, ylim1)
    plt.xlabel('Población 1')
    plt.ylabel('Población 2')
    plt.show()

def caso2():
    K1, K2, a12, a21 = casos['Caso 2']

    # Rectas de las isoclinas
    iso1 = K1 - a12 * y       # ordenada al origen --> K1 / a12       #raiz --> K1
    iso2 = K2 - a21 * x       # ordenada al origen --> K2             #raiz --> K2 / a21

    # Poblaciones en funcion del tiempo
    NP0_1 = np.array([25, 25])
    N1_1, N2_1 = runge_kutta(f, NP0_1, 'Caso 2')

    NP0_2 = np.array([50, 50])
    N1_2, N2_2 = runge_kutta(f, NP0_2, 'Caso 2')

    NP0_3 = np.array([100, 100])
    N1_3, N2_3 = runge_kutta(f, NP0_3, 'Caso 2')

    frac = 100
    
    plt.figure()
    plt.plot(iso1, y, label='Iso1')
    plt.plot(x, iso2, label='Iso2')
    plt.plot(N1_1, N2_1, color='black')

    plt.arrow(N1_1[len(N1_1)//frac], N2_1[len(N2_1)//frac], N1_1[len(N1_1)//frac + 1] - N1_1[len(N1_1)//frac], N2_1[len(N2_1)//frac + 1] - N2_1[len(N2_1)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_2, N2_2, color='black')

    plt.arrow(N1_2[len(N1_2)//frac], N2_2[len(N2_2)//frac], N1_2[len(N1_2)//frac + 1] - N1_2[len(N1_2)//frac], N2_2[len(N2_2)//frac + 1] - N2_2[len(N2_2)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_3, N2_3, color='black')

    plt.arrow(N1_3[len(N1_3)//frac], N2_3[len(N2_3)//frac], N1_3[len(N1_3)//frac + 1] - N1_3[len(N1_3)//frac], N2_3[len(N2_3)//frac + 1] - N2_3[len(N2_3)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.xlim(0, xlim1)
    plt.ylim(0, ylim1)
    plt.xlabel('Población 1')
    plt.ylabel('Población 2')
    
    plt.show()


def caso3():
    K1, K2, a12, a21 = casos['Caso 3']

    # Rectas de las isoclinas
    iso1 = K1 - a12 * y       # ordenada al origen --> K1 / a12       #raiz --> K1
    iso2 = K2 - a21 * x       # ordenada al origen --> K2             #raiz --> K2 / a21

    # Poblaciones en funcion del tiempo
    NP0_1 = np.array([25, 25])
    N1_1, N2_1 = runge_kutta(f, NP0_1, 'Caso 3')

    NP0_2 = np.array([75, 10])
    N1_2, N2_2 = runge_kutta(f, NP0_2, 'Caso 3')

    NP0_3 = np.array([175, 20])
    N1_3, N2_3 = runge_kutta(f, NP0_3, 'Caso 3')

    NP0_4 = np.array([200 - (1000 / 11), (200 / 11)])
    N1_4, N2_4 = runge_kutta(f, NP0_4, 'Caso 3')

    NP0_5 = np.array([100, 60])
    N1_5, N2_5 = runge_kutta(f, NP0_5, 'Caso 3')

    NP0_6 = np.array([25, 100])
    N1_6, N2_6 = runge_kutta(f, NP0_6, 'Caso 3')

    frac = 100
    
    plt.figure()
    plt.plot(iso1, y, label='Iso1')
    plt.plot(x, iso2, label='Iso2')
    
    plt.plot(N1_1, N2_1, color='black')
    plt.arrow(N1_1[len(N1_1)//frac], N2_1[len(N2_1)//frac], N1_1[len(N1_1)//frac + 1] - N1_1[len(N1_1)//frac], N2_1[len(N2_1)//frac + 1] - N2_1[len(N2_1)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_2, N2_2, color='black')
    plt.arrow(N1_2[len(N1_2)//frac], N2_2[len(N2_2)//frac], N1_2[len(N1_2)//frac + 1] - N1_2[len(N1_2)//frac], N2_2[len(N2_2)//frac + 1] - N2_2[len(N2_2)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_3, N2_3, color='black')
    plt.arrow(N1_3[len(N1_3)//frac], N2_3[len(N2_3)//frac], N1_3[len(N1_3)//frac + 1] - N1_3[len(N1_3)//frac], N2_3[len(N2_3)//frac + 1] - N2_3[len(N2_3)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_4, N2_4, color='black')
    plt.arrow(N1_4[len(N1_4)//frac], N2_4[len(N2_4)//frac], N1_4[len(N1_4)//frac + 1] - N1_4[len(N1_4)//frac], N2_4[len(N2_4)//frac + 1] - N2_4[len(N2_4)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_5, N2_5, color='black')
    plt.arrow(N1_5[len(N1_5)//frac], N2_5[len(N2_5)//frac], N1_5[len(N1_5)//frac + 1] - N1_5[len(N1_5)//frac], N2_5[len(N2_5)//frac + 1] - N2_5[len(N2_5)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_6, N2_6, color='black')
    plt.arrow(N1_6[len(N1_6)//frac], N2_6[len(N2_6)//frac], N1_6[len(N1_6)//frac + 1] - N1_6[len(N1_6)//frac], N2_6[len(N2_6)//frac + 1] - N2_6[len(N2_6)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.xlim(0, xlim1)
    plt.ylim(0, ylim1)
    plt.xlabel('Población 1')
    plt.ylabel('Población 2')
    
    plt.show()
    

def caso4():
    K1, K2, a12, a21 = casos['Caso 4']

    # Rectas de las isoclinas
    iso1 = K1 - a12 * y       # ordenada al origen --> K1 / a12       #raiz --> K1
    iso2 = K2 - a21 * x       # ordenada al origen --> K2             #raiz --> K2 / a21

    # Poblaciones en funcion del tiempo
    NP0_1 = np.array([25, 25])
    N1_1, N2_1 = runge_kutta(f, NP0_1, 'Caso 4')

    NP0_2 = np.array([75, 10])
    N1_2, N2_2 = runge_kutta(f, NP0_2, 'Caso 4')

    NP0_3 = np.array([150, 20])
    N1_3, N2_3 = runge_kutta(f, NP0_3, 'Caso 4')

    NP0_4 = np.array([150, 40])
    N1_4, N2_4 = runge_kutta(f, NP0_4, 'Caso 4')

    NP0_5 = np.array([100, 60])
    N1_5, N2_5 = runge_kutta(f, NP0_5, 'Caso 4')

    NP0_6 = np.array([25, 100])
    N1_6, N2_6 = runge_kutta(f, NP0_6, 'Caso 4')

    frac = 100
    
    plt.figure()
    plt.plot(iso1, y, label='Iso1')
    plt.plot(x, iso2, label='Iso2')
    
    plt.plot(N1_1, N2_1, color='black')
    plt.arrow(N1_1[len(N1_1)//frac], N2_1[len(N2_1)//frac], N1_1[len(N1_1)//frac + 1] - N1_1[len(N1_1)//frac], N2_1[len(N2_1)//frac + 1] - N2_1[len(N2_1)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_2, N2_2, color='black')
    plt.arrow(N1_2[len(N1_2)//frac], N2_2[len(N2_2)//frac], N1_2[len(N1_2)//frac + 1] - N1_2[len(N1_2)//frac], N2_2[len(N2_2)//frac + 1] - N2_2[len(N2_2)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_3, N2_3, color='black')
    plt.arrow(N1_3[len(N1_3)//frac], N2_3[len(N2_3)//frac], N1_3[len(N1_3)//frac + 1] - N1_3[len(N1_3)//frac], N2_3[len(N2_3)//frac + 1] - N2_3[len(N2_3)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_4, N2_4, color='black')
    plt.arrow(N1_4[len(N1_4)//frac], N2_4[len(N2_4)//frac], N1_4[len(N1_4)//frac + 1] - N1_4[len(N1_4)//frac], N2_4[len(N2_4)//frac + 1] - N2_4[len(N2_4)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_5, N2_5, color='black')
    plt.arrow(N1_5[len(N1_5)//frac], N2_5[len(N2_5)//frac], N1_5[len(N1_5)//frac + 1] - N1_5[len(N1_5)//frac], N2_5[len(N2_5)//frac + 1] - N2_5[len(N2_5)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')

    plt.plot(N1_6, N2_6, color='black')
    plt.arrow(N1_6[len(N1_6)//frac], N2_6[len(N2_6)//frac], N1_6[len(N1_6)//frac + 1] - N1_6[len(N1_6)//frac], N2_6[len(N2_6)//frac + 1] - N2_6[len(N2_6)//frac], head_width = 3, head_length = 4, fc = 'black', ec = 'black')
    

    plt.xlim(0, xlim1)
    plt.ylim(0, ylim1)
    plt.xlabel('Población 1')
    plt.ylabel('Población 2')
    
    plt.show()

caso1()
caso2()
caso3()
caso4()