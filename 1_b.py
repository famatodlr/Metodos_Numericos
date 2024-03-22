import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
from numpy import linspace
from numpy import meshgrid
from scipy.interpolate import griddata

def function(x1,x2):
    term1 = 0.75*np.exp(-(((10 *x1) -2)**2) / 4 - ((9*x2-2)**2) / 4)
    term2 = 0.65*np.exp(-((9*x1+1)**2)/9 -(((10 *x2 +1))**2)/2)
    term3 = 0.55*np.exp(-(9*x1 -6)**2/4 - ((9*x2 -3)**2) / 4)
    term4 = -0.01*np.exp((-(9 * x1 -7)**2) /4 - ((9*x2-7)**2)/4)
    return term1 + term2 + term3 + term4

x1 = linspace(0, 1, 100)
x2 = linspace(0, 1, 100)
n_points = 100
# Generar puntos para graficar la función de Franke
x1, x2 = np.meshgrid(x1, x2)
z = function(x1, x2)

# Puntos de interpolación para la funcion de Franke
x1_interpolation = np.random.rand(n_points)
x2_interpolation = np.random.rand(n_points)
z_interpolation = function(x1_interpolation, x2_interpolation)

# usar griddata
interpolacion = griddata((x1_interpolation, x2_interpolation), z_interpolation, (x1, x2), method='cubic')

# graficar la comparacion entre la funcion de franke y la interpolacion en 2 graficos 3d
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x1, x2, z, cmap='viridis')
ax.set_title('Funcion de Franke')
#nombrar los ejes
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(x1, x2, interpolacion, cmap='Oranges')
ax.set_title('Interpolacion')
#nombrar los ejes
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()


# comparacion del error de la interpolacion con la funcion de franke
z_real = function(x1, x2)
error = np.abs(z_real - z_interpolation)
plt.imshow(error, cmap='viridis')
plt.colorbar()
plt.title('Error de la interpolacion')
plt.show()

#error por coodenadas