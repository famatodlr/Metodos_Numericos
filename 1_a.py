import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
import numpy.polynomial.chebyshev
import math as m

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

# Errores relativos
error_lagrange = np.abs((y_values - y_lagrange) / y_values)
error_spline = np.abs((y_values - y_spline) / y_values)

# Graficar
def graficos():
    plt.figure(figsize=(12, 6))

    # Función original y Lagrage
    plt.subplot(2, 2, 1)
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, label='Puntos de interpolación', color='black')
    plt.plot(x_values, y_lagrange, label='Interpolación de Lagrange', linestyle='--', color='red')
    plt.title('Comparación entre fa(x), Interpolación de Lagrange')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)


    # Error relativo de Lagrange
    plt.subplot(2, 2, 2)
    plt.plot(x_values, error_lagrange, label='Error relativo de Lagrange', color='red')
    plt.title('Error relativo de Lagrange')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error relativo', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    # Función original y Spline
    plt.subplot(2, 2, 3)
    plt.plot(x_values, y_spline, label='Polinomio cúbico spline', linestyle='-.', color='green')
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, label='Puntos de interpolación', color='black')
    plt.title('Comparación entre fa(x) y Polinomi cúbico spline')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    #Error relativo de Spline
    plt.subplot(2, 2, 4)
    plt.title('Error relativo de Spline')
    plt.plot(x_values, error_spline, label='Error relativo del polinomio cúbico spline', color='green')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error relativo', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def get_chevichev_roots(n, a, b) -> list[int]:
    """
    Get the chevichev roots
    n: number of roots
    a: start of the interval (included)
    b: end of the interval (included)
    """
    roots = []
    for k in range(n - 1, 1, -1):
        roots.append((a+b)/2 + (b-a)/2 * m.cos(((2*k-1)/(2*n)) * m.pi))

    return (roots)

x_no_equispaciados = np.array(get_chevichev_roots(n_interpolation_points, a, b))
y_no_equispaciados = fa(x_no_equispaciados)

lagrange_poly_no_equispaciados = lagrange(x_no_equispaciados, y_no_equispaciados)
y_lagrange_no_equispaciados = lagrange_poly_no_equispaciados(x_values)

spline_no_equispaciados = CubicSpline(x_no_equispaciados, y_no_equispaciados)
y_spline_no_equispaciados = spline_no_equispaciados(x_values)

error_lagrange_no_equispaciados = np.abs((y_values - y_lagrange_no_equispaciados) / y_values)
error_spline_no_equispaciados = np.abs((y_values - y_spline_no_equispaciados) / y_values)

def graficos_no_equispaciados():
    plt.figure(figsize=(12, 6))

    # Función original y Lagrage
    plt.subplot(2, 2, 1)
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_no_equispaciados, y_no_equispaciados, label='Puntos de interpolación', color='black')
    plt.plot(x_values, y_lagrange_no_equispaciados, label='Interpolación de Lagrange', linestyle='--', color='red')
    plt.title('Comparación entre fa(x), Interpolación de Lagrange')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)


    # Error relativo de Lagrange
    plt.subplot(2, 2, 2)
    plt.plot(x_values, error_lagrange_no_equispaciados, label='Error relativo de Lagrange', color='red')
    plt.title('Error relativo de Lagrange')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error relativo', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    # Función original y Spline
    plt.subplot(2, 2, 3)
    plt.plot(x_values, y_spline_no_equispaciados, label='Polinomio cúbico spline', linestyle='-.', color='green')
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_no_equispaciados, y_no_equispaciados, label='Puntos de interpolación', color='black')
    plt.title('Comparación entre fa(x) y Polinomi cúbico spline')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    #Error relativo de Spline
    plt.subplot(2, 2, 4)
    plt.title('Error relativo de Spline')
    plt.plot(x_values, error_spline_no_equispaciados, label='Error relativo del polinomio cúbico', color='green')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error relativo', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

graficos()
graficos_no_equispaciados()