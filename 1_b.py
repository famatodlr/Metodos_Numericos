import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linspace
from numpy import meshgrid
from scipy.interpolate import griddata
import math as m

def function(x1,x2):
    term1 = 0.75*np.exp(-(((10 *x1) -2)**2) / 4 - ((9*x2-2)**2) / 4)
    term2 = 0.65*np.exp(-((9*x1+1)**2)/9 -(((10 *x2 +1))**2)/2)
    term3 = 0.55*np.exp(-(9*x1 -6)**2/4 - ((9*x2 -3)**2) / 4)
    term4 = -0.01*np.exp((-(9 * x1 -7)**2) /4 - ((9*x2-7)**2)/4)
    return term1 + term2 + term3 + term4

# Definir el intervalo
a = -1
b = 1
n_points = 1000  # Número de puntos para graficar fa(x)

# Generar puntos para graficar la función fa(x)
x1 = np.linspace(a, b, n_points)
x2 = np.linspace(a, b, n_points)
X1, X2 = meshgrid(x1, x2)
Y = function(X1, X2)

# Puntos de interpolación equispaciados
n_interpolation_points = 10
x_interpolation = np.linspace(a, b, n_interpolation_points)
y_interpolation = function(x_interpolation, x_interpolation)

X1_int, X2_int = meshgrid(x_interpolation, x_interpolation)
points = np.column_stack((X1_int.ravel(), X2_int.ravel()))
values = function(X1_int, X2_int).ravel()

# Interpolación en 3d
interpolated_points = griddata(points, values, (X1, X2), method='cubic')

# Errores absolutos
error = np.abs(Y - interpolated_points)

# Graficar
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='viridis')
ax.set_title('Función original')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.view_init(30, 30)

# graficar los puntos de interpolación
ax.scatter(X1_int, X2_int, function(X1_int, X2_int), color='red')

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X1, X2, interpolated_points, cmap='viridis')
ax.set_title('Interpolación')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.show()

# Graficar el error absoluto en heatmap 
plt.figure(figsize=(12, 6))
plt.imshow(error, cmap='hot', extent=[a, b, a, b])
plt.colorbar()
plt.title('Error absoluto')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Puntos de interpolacion no equispaciados

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
y_no_equispaciados = function(x_no_equispaciados, x_no_equispaciados)

X1_int, X2_int = meshgrid(x_no_equispaciados, x_no_equispaciados)
points = np.column_stack((X1_int.ravel(), X2_int.ravel()))
values = function(X1_int, X2_int).ravel()

# Interpolación en 3d
interpolated_points = griddata(points, values, (X1, X2), method='cubic')

# Errores absolutos
error = np.abs(Y - interpolated_points)

# Graficar
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='viridis')
ax.set_title('Función original')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.view_init(30, 30)

# graficar los puntos de interpolación
ax.scatter(X1_int, X2_int, function(X1_int, X2_int), color='red')

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X1, X2, interpolated_points, cmap='viridis')
ax.set_title('Interpolación')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.show()

# Graficar el error absoluto en heatmap
plt.figure(figsize=(12, 6))
plt.imshow(error, cmap='hot', extent=[a, b, a, b])
plt.colorbar()
plt.title('Error absoluto')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
