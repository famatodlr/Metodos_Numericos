import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
import numpy.polynomial.chebyshev

# Definir la función fa(x)
def fa(x):
    return 0.3**np.abs(x) * np.sin(4*x) - np.tanh(2*x) + 2

# Definir el intervalo
a = -4
b = 4
n_points = 1000  # Número de puntos para graficar fa(x)

# Generar puntos para graficar la función fa(x)
x_values = np.linspace(a, b, n_points)
y_values = fa(x_values)

# Puntos de interpolación
n_interpolation_points = 10
x_interpolation = np.linspace(a, b, n_interpolation_points)
y_interpolation = fa(x_interpolation)

# Interpolación de Lagrange
lagrange_poly = lagrange(x_interpolation, y_interpolation)
y_lagrange = lagrange_poly(x_values)

# Polinomios cúbicos splines
spline = CubicSpline(x_interpolation, y_interpolation)
y_spline = spline(x_values)

# Polinomios de Chebyshev
def chebyshev_polynomial_interpolation(n, a, b):
    x = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))
    y = fa((b - a) / 2 * x + (a + b) / 2)
    c = numpy.polynomial.chebyshev.chebfit(x, y, n - 1)
    return numpy.polynomial.chebyshev.Chebyshev(c, domain=[a, b])

n_chebyshev = 10
chebyshev_poly = chebyshev_polynomial_interpolation(n_chebyshev, a, b)
y_chebyshev = chebyshev_poly(x_values)
chebroots = chebyshev_poly.roots()

# Errores relativos
error_lagrange = np.abs((y_values - y_lagrange) / y_values)
error_spline = np.abs((y_values - y_spline) / y_values)
error_chebyshev = np.abs((y_values - y_chebyshev) / y_values)

# Graficar
def graficos():
    plt.figure(figsize=(12, 6))
    #quiero hacer que los graficos tengan relacion 1:1, como hago eso?
    # plt.axis('equal')

    # Función original y Lagrage
    plt.subplot(3, 2, 1)
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, label='Puntos de interpolación', color='black')
    plt.plot(x_values, y_lagrange, label='Interpolación de Lagrange', linestyle='--', color='red')
    plt.title('Comparación entre fa(x), Interpolación de Lagrange')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)


    # Error relativo de Lagrange
    plt.subplot(3, 2, 2)
    plt.plot(x_values, error_lagrange, label='Error relativo de Lagrange', color='red')
    plt.title('Error relativo de Lagrange')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error relativo', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    # Función original y Spline
    plt.subplot(3, 2, 3)
    plt.plot(x_values, y_spline, label='Polinomio cúbico spline', linestyle='-.', color='green')
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, label='Puntos de interpolación', color='black')
    plt.title('Comparación entre fa(x) y Polinomi cúbico spline')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    #Error relativo de Spline
    plt.subplot(3, 2, 4)
    plt.title('Error relativo de Spline')
    plt.plot(x_values, error_spline, label='Error relativo del polinomio cúbico spline', color='green')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error relativo', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    # Función original y Chebyshev
    plt.subplot(3, 2, 5)
    plt.plot(x_values, y_chebyshev, label='Polinomio de Chebyshev', linestyle=':', color='purple')
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, label='Puntos de interpolación', color='black')
    plt.title('Comparación entre fa(x) y Polinomio de Chebyshev')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    #Error relativo de Chebyshev
    plt.subplot(3, 2, 6)
    plt.title('Error relativo de Chebyshev')
    plt.plot(x_values, error_chebyshev, label='Error relativo del polinomio de Chebyshev', color='purple')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error relativo', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()

graficos()
