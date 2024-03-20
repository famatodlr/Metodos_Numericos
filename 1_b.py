import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
import numpy.polynomial.chebyshev
from numpy import exp

#tengo que analizar lagrange y splines en una funcion de dos variables
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
n_points = 100
# Generar puntos para graficar la función de Franke
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

# Puntos de interpolación para la funcion de Franke
x_interpolation = np.random.rand(n_points)
y_interpolation = np.random.rand(n_points)
z_interpolation = FrankeFunction(x_interpolation, y_interpolation)

# Interpolación de Lagrange para la función de Franke
lagrange_poly = lagrange((x_interpolation, y_interpolation), z_interpolation)
z_lagrange = lagrange_poly((x, y))
# Polinomios cúbicos splines para la función de Franke
spline = CubicSpline((x_interpolation, y_interpolation), z_interpolation)
z_spline = spline((x, y))


# Polinomios de Chebyshev para la función de Franke
def chebyshev_polynomial_interpolation(n, a, b):
    x = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))
    y = FrankeFunction((b - a) / 2 * x + (a + b) / 2)
    c = numpy.polynomial.chebyshev.chebfit(x, y, n - 1)
    return numpy.polynomial.chebyshev.Chebyshev(c, domain=[a, b])

n_chebyshev = 10
chebyshev_poly = chebyshev_polynomial_interpolation(n_chebyshev, 0, 1)
z_chebyshev = chebyshev_poly(x, y)
chebroots = chebyshev_poly.roots()

# Errores relativos para la función de Franke
error_lagrange = np.abs((z - z_lagrange) / z)
error_spline = np.abs((z - z_spline) / z)
error_chebyshev = np.abs((z - z_chebyshev) / z)

# Graficar para la función de Franke
def graficos():
    plt.figure(figsize=(12, 6))
    #quiero hacer que los graficos tengan relacion 1:1, como hago eso?
    # plt.axis('equal')

    # Función original y Lagrage
    plt.subplot(3, 2, 1)
    plt.plot(x, y, z, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, z_interpolation, label='Puntos de interpolación', color='black')
    plt.plot(x, y, z_lagrange, label='Interpolación de Lagrange', linestyle='--', color='red')
    plt.title('Comparación entre fa(x), Interpolación de Lagrange')
    plt.legend()
    plt.grid(True)


    # Error relativo de Lagrange para la función de Franke
    plt.subplot(3, 2, 2)
    plt.plot(x, y, error_lagrange, label='Error relativo de Lagrange', color='red')
    plt.title('Error relativo de Lagrange')
    plt.legend()
    plt.grid(True)

    # Función original y Spline
    plt.subplot(3, 2, 3)
    plt.plot(x, y, z, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, z_interpolation, label='Puntos de interpolación', color='black')
    plt.plot(x, y, z_spline, label='Polinomio cúbico spline', linestyle=':', color='green')

    plt.title('Comparación entre fa(x) y Polinomi cúbico spline')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    #Error relativo de Spline para la función de Franke
    plt.subplot(3, 2, 4)
    plt.title('Error relativo de Spline')
    plt.plot(x, y, error_spline, label='Error relativo del polinomio cúbico spline', color='green')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error relativo', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    # Función original y Chebyshev
    plt.subplot(3, 2, 5)
    plt.plot(x, y, z, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, z_interpolation, label='Puntos de interpolación', color='black')
    plt.plot(x, y, z_chebyshev, label='Polinomio de Chebyshev', linestyle='-.', color='purple')
    plt.title('Comparación entre fa(x) y Polinomio de Chebyshev')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    #Error relativo de Chebyshev para la función de Franke
    plt.subplot(3, 2, 6)
    plt.title('Error relativo de Chebyshev')
    plt.plot(x, y, error_chebyshev, label='Error relativo del polinomio de Chebyshev', color='purple')   
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error relativo', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()

graficos()
