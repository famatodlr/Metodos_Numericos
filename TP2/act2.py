#quiero hacer runge kutta para un sistema de odes de orden 1
import numpy as np
import matplotlib.pyplot as plt
#importar el runge kutta 4 de scipy
from scipy.integrate import solve_ivp


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
# t = np.linspace(0, 200, 1000)
# n1 = []
# n2 = []
# for i in t:
#     n1_, n2_ = runge_kutta_2(n1_0, n2_0, r1, K1, a1, r2, K2, a2, 0.1, int(i))
#     n1.append(n1_)
#     n2.append(n2_)
# plt.plot(t, n1, label='Especie 1')
# plt.plot(t, n2, label='Especie 2')
# plt.xlabel('Tiempo')
# plt.ylabel('Poblacion')
# plt.title('Modelo de Competencia de Lotka-Volterra')
# plt.legend()
# #label de las condiciones iniciales
# plt.annotate(f'Condiciones iniciales: n1 = {n1_0}, n2 = {n2_0}\n, K1 = {K1}, K2 = {K2}\n, r1 = {r1}, r2 = {r2}\n, a1 = {a1}, a2 = {a2}',
             
#               xy=(0, 1), xycoords='axes fraction', fontsize=8,
#              xytext=(10, -10), textcoords='offset points',
#              ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

# plt.show()


''' Consideremos que en el sistema ahora hay dos especies que compiten por los mismos recursos. La ecua-
ción logística ya considera la existencia de competencia intraespecífica, debido a que los recursos se trans-
forman en un limitante a medida que la población se incrementa. Sin embargo, la ecuación logística 4 no es
suficiente si tenemos dos especies en el sistema. Es necesario modelar cómo las dos especies se interrelacio-
nan estableciendo una competencia entre ellas por los recursos (competencia interespecífica). La forma más
sencilla de modelar la competencia interespecífica es mediante un término que reduzca la capacidad de carga
y sea proporcional a la cantidad de individuos de cada especie. De esta forma se definen las ecuaciones de
competencia de Lotka-Volterra'''

def especie1(r1, n1, K1, a1, n2, K2):
        return r1 * n1 * ((K1 - n1 - a1 * n2) / K1)

def especie2(r2, n2, K2, a2, n1, K1):
        return r2 * n2 * ((K2 - n2 - a2 * n1) / K2)

def runge_kutta_4(n1, n2, r1, K1, a1, r2, K2, a2, h, t):
    n1 = n1
    n2 = n2
    for _ in range(t):
        k1 = h * especie1(r1, n1, K1, a1, n2, K2)
        l1 = h * especie2(r2, n2, K2, a2, n1, K1)
        
        k2 = h * especie1(r1, n1 + 0.5 * k1, K1, a1, n2 + 0.5 * l1, K2)
        l2 = h * especie2(r2, n2 + 0.5 * l1, K2, a2, n1 + 0.5 * k1, K1)

        k3 = h * especie1(r1, n1 + 0.5 * k2, K1, a1, n2 + 0.5 * l2, K2)
        l3 = h * especie2(r2, n2 + 0.5 * l2, K2, a2, n1 + 0.5 * k2, K1)

        k4 = h * especie1(r1, n1 + k3, K1, a1, n2 + l3, K2)
        l4 = h * especie2(r2, n2 + l3, K2, a2, n1 + k3, K1)

        n1 = n1 + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        n2 = n2 + (1/6) * (l1 + 2*l2 + 2*l3 + l4)
    return n1, n2
def graficos():
        
    def grafico1():
        datos_0 = {'Especie1_0': 100, 'Especie2_0': 100, 'K1': 50, 'K2': 25, 'r1': 0, 'r2': 0.1, 'a1': 0, 'a2': 0}
        t = np.linspace(0, 200, 1000)
        n1 = []
        n2 = []
        for i in t:
            n1_, n2_ = runge_kutta_4(datos_0['Especie1_0'], datos_0['Especie2_0'], datos_0['r1'], datos_0['K1'], datos_0['a1'], datos_0['r2'], datos_0['K2'], datos_0['a2'], 0.1, int(i))   
            n1.append(n1_)
            n2.append(n2_)
        plt.plot(t, n1, label='Especie 1')
        plt.plot(t, n2, label='Especie 2')
        plt.xlabel('Tiempo')
        plt.ylabel('Poblacion')
        plt.title('Modelo de Competencia de Lotka-Volterra')
        plt.legend()
        #label de las condiciones iniciales
        plt.annotate(f'Condiciones iniciales:\nn1 = {datos_0["Especie1_0"]}, n2 = {datos_0["Especie2_0"]}\n, K1 = {datos_0["K1"]}, K2 = {datos_0["K2"]}\n, r1 = {datos_0["r1"]}, r2 = {datos_0["r2"]}\n, a1 = {datos_0["a1"]}, a2 = {datos_0["a2"]}',
                    xy=(0, 1), xycoords='axes fraction', fontsize=6,
                    xytext=(10, -10), textcoords='offset points',
                    ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.show()

    #caso 1: poblaciones iniciales iguales a cero
    def grafico2():
        datos_1 = {'Especie1_0': 1, 'Especie2_0': 1, 'K1': 50, 'K2': 50, 'r1': 0.1, 'r2': 0.1, 'a1': 0, 'a2': 0}
        t = np.linspace(0, 5000, 1000)
        n1 = []
        n2 = []
        for i in t:
            n1_, n2_ = runge_kutta_4(datos_1['Especie1_0'], datos_1['Especie2_0'], datos_1['r1'], datos_1['K1'], datos_1['a1'], datos_1['r2'], datos_1['K2'], datos_1['a2'], 0.1, int(i))   
            n1.append(n1_)
            n2.append(n2_)
        plt.plot(t, n1, label='Especie 1')
        plt.plot(t, n2, label='Especie 2')
        plt.xlabel('Tiempo')
        plt.ylabel('Poblacion')
        plt.title('Modelo de Competencia de Lotka-Volterra')
        plt.legend()
        #label de las condiciones iniciales
        plt.annotate(f'Condiciones iniciales: n1 = {datos_1["Especie1_0"]}, n2 = {datos_1["Especie2_0"]}\n, K1 = {datos_1["K1"]}, K2 = {datos_1["K2"]}\n, r1 = {datos_1["r1"]}, r2 = {datos_1["r2"]}\n, a1 = {datos_1["a1"]}, a2 = {datos_1["a2"]}',
                    xy=(0, 1), xycoords='axes fraction', fontsize=6,
                    xytext=(10, -10), textcoords='offset points',
                    ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.show()

    #caso 2: poblaciones iniciales iguales a las capacidades de carga
    def grafico3():
        datos_2 = {'Especie1_0': 50, 'Especie2_0': 25, 'K1': 50, 'K2': 25, 'r1': 0.1, 'r2': 0.1, 'a1': 0, 'a2': 0}
        t = np.linspace(0, 5000, 1000)
        n1 = []
        n2 = []
        for i in t:
            n1_, n2_ = runge_kutta_4(datos_2['Especie1_0'], datos_2['Especie2_0'], datos_2['r1'], datos_2['K1'], datos_2['a1'], datos_2['r2'], datos_2['K2'], datos_2['a2'], 0.1, int(i))   
            n1.append(n1_)
            n2.append(n2_)
        plt.plot(t, n1, label='Especie 1')
        plt.plot(t, n2, label='Especie 2')
        plt.xlabel('Tiempo')
        plt.ylabel('Poblacion')
        plt.title('Modelo de Competencia de Lotka-Volterra')
        plt.legend()
        #label de las condiciones iniciales
        plt.annotate(f'Condiciones iniciales: n1 = {datos_2["Especie1_0"]}, n2 = {datos_2["Especie2_0"]}\n, K1 = {datos_2["K1"]}, K2 = {datos_2["K2"]}\n, r1 = {datos_2["r1"]}, r2 = {datos_2["r2"]}\n, a1 = {datos_2["a1"]}, a2 = {datos_2["a2"]}',
                    xy=(0, 1), xycoords='axes fraction', fontsize=6,
                    xytext=(10, -10), textcoords='offset points',
                    ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.show()

    #caso 3: competencia intraespecífica, pero una tiene una competencia mayor con la otra
    def grafico4():
        datos_3 = {'Especie1_0': 100, 'Especie2_0': 100, 'K1': 500, 'K2': 500, 'r1': 0.1, 'r2': 0.1, 'a1': 0.4, 'a2': 0.1}
        t = np.linspace(0, 5000, 1000)
        n1 = []
        n2 = []
        for i in t:
            n1_, n2_ = runge_kutta_4(datos_3['Especie1_0'], datos_3['Especie2_0'], datos_3['r1'], datos_3['K1'], datos_3['a1'], datos_3['r2'], datos_3['K2'], datos_3['a2'], 0.1, int(i))   
            n1.append(n1_)
            n2.append(n2_)
        plt.plot(t, n1, label='Especie 1')
        plt.plot(t, n2, label='Especie 2')
        plt.xlabel('Tiempo')
        plt.ylabel('Poblacion')
        plt.title('Modelo de Competencia de Lotka-Volterra')
        plt.legend()
        #label de las condiciones iniciales
        plt.annotate(f'Condiciones iniciales: n1 = {datos_3["Especie1_0"]}, n2 = {datos_3["Especie2_0"]}\n, K1 = {datos_3["K1"]}, K2 = {datos_3["K2"]}\n, r1 = {datos_3["r1"]}, r2 = {datos_3["r2"]}\n, a1 = {datos_3["a1"]}, a2 = {datos_3["a2"]}',
                    xy=(0, 1), xycoords='axes fraction', fontsize=6,
                    xytext=(10, -10), textcoords='offset points',
                    ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.show()

    #caso 4: competencia intraespecífica, pero facilitada
    def grafico5():
        datos_4 = {'Especie1_0': 100, 'Especie2_0': 100, 'K1': 500, 'K2': 500, 'r1': 0.1, 'r2': 0.1, 'a1': -0.1, 'a2': -0.1}
        t = np.linspace(0, 5000, 500)
        n1 = []
        n2 = []
        for i in t:
            n1_, n2_ = runge_kutta_4(datos_4['Especie1_0'], datos_4['Especie2_0'], datos_4['r1'], datos_4['K1'], datos_4['a1'], datos_4['r2'], datos_4['K2'], datos_4['a2'], 0.1, int(i))   
            n1.append(n1_)
            n2.append(n2_)
        plt.plot(t, n1, label='Especie 1')
        plt.plot(t, n2, label='Especie 2')
        plt.xlabel('Tiempo')
        plt.ylabel('Poblacion')
        plt.title('Modelo de Competencia de Lotka-Volterra')
        plt.legend()
        #label de las condiciones iniciales
        plt.annotate(f'Condiciones iniciales: n1 = {datos_4["Especie1_0"]}, n2 = {datos_4["Especie2_0"]}\n, K1 = {datos_4["K1"]}, K2 = {datos_4["K2"]}\n, r1 = {datos_4["r1"]}, r2 = {datos_4["r2"]}\n, a1 = {datos_4["a1"]}, a2 = {datos_4["a2"]}',
                    xy=(0, 1), xycoords='axes fraction', fontsize=6,
                    xytext=(10, -10), textcoords='offset points',
                    ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.show()

    #caso 5: especie dominante (tasa de crecimiento mayor)
    def grafico6():
        datos_5 = {'Especie1_0': 100, 'Especie2_0': 100, 'K1': 500, 'K2': 500, 'r1': 0.9, 'r2': 0.1, 'a1': 0.1, 'a2': 0.1}
        t = np.linspace(0, 5000, 500)
        n1 = []
        n2 = []
        for i in t:
            n1_, n2_ = runge_kutta_4(datos_5['Especie1_0'], datos_5['Especie2_0'], datos_5['r1'], datos_5['K1'], datos_5['a1'], datos_5['r2'], datos_5['K2'], datos_5['a2'], 0.1, int(i))   
            n1.append(n1_)
            n2.append(n2_)
        plt.plot(t, n1, label='Especie 1')
        plt.plot(t, n2, label='Especie 2')
        plt.xlabel('Tiempo')
        plt.ylabel('Poblacion')
        plt.title('Modelo de Competencia de Lotka-Volterra')
        plt.legend()
        #label de las condiciones iniciales
        plt.annotate(f'Condiciones iniciales: n1 = {datos_5["Especie1_0"]}, n2 = {datos_5["Especie2_0"]}\n, K1 = {datos_5["K1"]}, K2 = {datos_5["K2"]}\n, r1 = {datos_5["r1"]}, r2 = {datos_5["r2"]}\n, a1 = {datos_5["a1"]}, a2 = {datos_5["a2"]}',
                    xy=(0, 1), xycoords='axes fraction', fontsize=6,
                    xytext=(10, -10), textcoords='offset points',
                    ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.show()

    #caso 6: especie dominante (mayor capacidad de carga)
    def grafico7():
        datos_6 = {'Especie1_0': 100, 'Especie2_0': 100, 'K1': 500, 'K2': 250, 'r1': 0.1, 'r2': 0.1, 'a1': 0.1, 'a2': 0.1}
        t = np.linspace(0, 500, 1000)
        n1 = []
        n2 = []
        for i in t:
            n1_, n2_ = runge_kutta_4(datos_6['Especie1_0'], datos_6['Especie2_0'], datos_6['r1'], datos_6['K1'], datos_6['a1'], datos_6['r2'], datos_6['K2'], datos_6['a2'], 0.1, int(i))   
            n1.append(n1_)
            n2.append(n2_)
        plt.plot(t, n1, label='Especie 1')
        plt.plot(t, n2, label='Especie 2')
        plt.xlabel('Tiempo')
        plt.ylabel('Poblacion')
        plt.title('Modelo de Competencia de Lotka-Volterra')
        plt.legend()
        #label de las condiciones iniciales
        plt.annotate(f'Condiciones iniciales: n1 = {datos_6["Especie1_0"]}, n2 = {datos_6["Especie2_0"]}\n, K1 = {datos_6["K1"]}, K2 = {datos_6["K2"]}\n, r1 = {datos_6["r1"]}, r2 = {datos_6["r2"]}\n, a1 = {datos_6["a1"]}, a2 = {datos_6["a2"]}',
                    xy=(0, 1), xycoords='axes fraction', fontsize=6,
                    xytext=(10, -10), textcoords='offset points',
                    ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.show()

#calculo de las curvas isoclinas cero
#como se calculan? se igualan a cero las ecuaciones de las especies
#para la especie 1:
# r1 * n1 * ((K1 - n1 - a1 * n2) / K1) = 0
# n1 = 0 o n1 = K1 - a1 * n2

#para la especie 2:
# r2 * n2 * ((K2 - n2 - a2 * n1) / K2) = 0
# n2 = 0 o n2 = K2 - a2 * n1

#grafico de las curvas isoclinas cero 
def grafico_isoclinas():
    datos = {'K1': 500, 'K2': 500, 'a1': 0.1, 'a2': 0.1}
    n1 = np.linspace(0, datos['K1'], 1000)
    n2 = np.linspace(0, datos['K2'], 1000)
    n1_, n2_ = np.meshgrid(n1, n2)
    dndt1 = especie1(0.1, n1_, datos['K1'], datos['a1'], n2_, datos['K2'])
    dndt2 = especie2(0.1, n2_, datos['K2'], datos['a2'], n1_, datos['K1'])
    plt.contour(n1, n2, dndt1, levels=[0], colors='red')
    plt.contour(n1, n2, dndt2, levels=[0], colors='blue')
    plt.xlabel('Especie 1')
    plt.ylabel('Especie 2')
    plt.title('Curvas Isoclinas')
    plt.show()

#grafico de las trayectorias, con intesidad de color segun la poblacion REVISAR
def grafico_trayectorias():
    datos = {'K1': 50, 'K2': 25, 'a1': 0, 'a2': 0}
    n1 = np.linspace(0, datos['K1'], 1000)
    n2 = np.linspace(0, datos['K2'], 1000)
    n1_, n2_ = np.meshgrid(n1, n2)
    dndt1 = especie1(0.1, n1_, datos['K1'], datos['a1'], n2_, datos['K2'])
    dndt2 = especie2(0.1, n2_, datos['K2'], datos['a2'], n1_, datos['K1'])
    plt.streamplot(n1, n2, dndt1, dndt2, color=np.sqrt(dndt1**2 + dndt2**2), linewidth=2, cmap='viridis')
 
    plt.colorbar()

    plt.xlabel('Especie 1')
    plt.ylabel('Especie 2')
    plt.title('Trayectorias')
    plt.show()
    

if __name__ == '__main__':
    # grafico_isoclinas()
    grafico_trayectorias()