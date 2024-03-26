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

#interpolacion lineal
linear_interpolation = np.interp(x_values, x_interpolation, y_interpolation)


# Interpolación de Lagrange
lagrange_poly = lagrange(x_interpolation, y_interpolation)
y_lagrange = lagrange_poly(x_values)

# Polinomios cúbicos splines
spline = CubicSpline(x_interpolation, y_interpolation)
y_spline = spline(x_values)

# Errores absolutos
error_lagrange = np.abs(y_values - y_lagrange)
error_spline = np.abs((y_values - y_spline))
error_linear = np.abs((y_values - linear_interpolation))

    


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

error_lagrange_no_equispaciados = np.abs((y_values - y_lagrange_no_equispaciados))
error_spline_no_equispaciados = np.abs((y_values - y_spline_no_equispaciados))

def grafico_lagrange():
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, label='Puntos de interpolación', color='black')
    plt.plot(x_values, y_lagrange, label='Interpolación de Lagrange', linestyle='--', color='red')
    plt.title('Comparación entre fa(x) y la interpolación de Lagrange con puntos equiespaciados')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)
    plt.show()

def grafico_lagrange_no_equispaciados():
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_no_equispaciados, y_no_equispaciados, label='Puntos de interpolación', color='black')
    plt.plot(x_values, y_lagrange_no_equispaciados, label='Interpolación de Lagrange', linestyle='--', color='red')
    plt.title('Comparación entre fa(x) y la interpolación de Lagrange con puntos no equiespaciados')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)
    plt.show()

def comparacion_errores_lagrange():
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, error_lagrange, label='Error absoluto de Lagrange', color='red')
    plt.plot(x_values, error_lagrange_no_equispaciados, label='Error absoluto de Lagrange con puntos no equiespaciados', color='green')
    plt.title('Error absoluto de Lagrange')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error absoluto', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)
    plt.show()

def grafico_spline():
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, label='Puntos de interpolación', color='black')
    plt.plot(x_values, y_spline, label='Polinomio cúbico spline', linestyle='-.', color='green')
    plt.title('Comparación entre fa(x) y el polinomio cúbico spline')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)
    plt.show()

def grafico_spline_no_equispaciados():
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_no_equispaciados, y_no_equispaciados, label='Puntos de interpolación', color='black')
    plt.plot(x_values, y_spline_no_equispaciados, label='Polinomio cúbico spline', linestyle='-.', color='green')
    plt.title('Comparación entre fa(x) y el polinomio cúbico spline con puntos no equiespaciados')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)
    plt.show()

def comparacion_errores_spline():
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, error_spline, label='Error absoluto del polinomio cúbico', color='green')
    plt.plot(x_values, error_spline_no_equispaciados, label='Error absoluto del polinomio cúbico con puntos no equiespaciados', color='red')
    plt.title('Error absoluto del polinomio cúbico spline')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error absoluto', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)
    plt.show()   
    




def graficos():
# Graficar en un mismo gráfico la función fa(x), la interpolación de Lagrange y el polinomio cúbico spline y la interpolacion lineal
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_interpolation, y_interpolation, label='Puntos de interpolación', color='black')
    plt.plot(x_values, y_spline, label='Polinomio cúbico spline', linestyle='-.', color='green')
    plt.plot(x_values, linear_interpolation, label='Interpolación lineal', linestyle=':', color='orange')
    plt.title('Comparación entre fa(x), Interpolación Lineal y Polinomio cúbico spline')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('y', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graficar el error absoluto de la interpolación de Lagrange y el polinomio cúbico spline
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, error_lagrange, label='Error absoluto de Lagrange', color='red')
    plt.plot(x_values, error_spline, label='Error absoluto del polinomio cúbico', color='green')
    plt.title('Error absoluto de Lagrange y Spline')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error absoluto', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)
    plt.show()

def graficos_no_equispaciados():
    plt.figure(figsize=(12, 6))

    # Función original y Lagrage
    plt.subplot(2, 2, 1)
    plt.plot(x_values, y_values, label='fa(x)', color='blue')
    plt.scatter(x_no_equispaciados, y_no_equispaciados, label='Puntos de interpolación', color='black')
    plt.plot(x_values, y_lagrange_no_equispaciados, label='Interpolación de Lagrange', linestyle='--', color='red')
    plt.title('Comparación entre fa(x), Interpolación de Lagrange con Puntos no equiespaciados')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('fa(x)', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)


    # Error absoluto de Lagrange
    plt.subplot(2, 2, 2)
    plt.plot(x_values, error_lagrange_no_equispaciados, label='Error absoluto de Lagrange', color='red')
    plt.title('Error absoluto de Lagrange')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error absoluto', fontweight='bold', loc='top')
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

    #Error absoluto de Spline
    plt.subplot(2, 2, 4)
    plt.title('Error absoluto de Spline')
    plt.plot(x_values, error_spline_no_equispaciados, label='Error absoluto del polinomio cúbico', color='green')
    plt.xlabel('x', fontweight='bold', loc='right')
    plt.ylabel('Error absoluto', fontweight='bold', loc='top')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


grafico_lagrange()
grafico_lagrange_no_equispaciados()
comparacion_errores_lagrange()
grafico_spline()
grafico_spline_no_equispaciados()
comparacion_errores_spline()
