import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline

x = []
y = []

with open("mnyo_mediciones.csv", 'r') as file:
    data = file.read()

    data = data.split("\n")
    data.remove(data[-1])

    for i in range(len(data)):
        data[i] = data[i].split(" ")

    for i in range(len(data)):
        for ii in range(len(data[i])):
            data[i][ii] = data[i][ii].split("e")

    for i in range(len(data)):
        for ii in range(len(data[i])):
            data[i][ii][0] = float(data[i][ii][0])
            data[i][ii][1] = int(data[i][ii][1])

    for i in range(len(data)):
        x.append(data[i][0][0] * (10 ** data[i][0][1]))
        y.append(data[i][1][0] * (10 ** data[i][1][1]))

real_x = []
real_y = []

with open("mnyo_ground_truth.csv", 'r') as file:
    data = file.read()

    data = data.split("\n")
    data.remove(data[-1])

    for i in range(len(data)):
        data[i] = data[i].split(" ")

    for i in range(len(data)):
        for ii in range(len(data[i])):
            data[i][ii] = data[i][ii].split("e")

    for i in range(len(data)):
        for ii in range(len(data[i])):
            data[i][ii][0] = float(data[i][ii][0])
            data[i][ii][1] = int(data[i][ii][1])

    for i in range(len(data)):
        real_x.append(data[i][0][0] * (10 ** data[i][0][1]))
        real_y.append(data[i][1][0] * (10 ** data[i][1][1]))

# spline = CubicSpline(x, y)
# y_spline = spline(x)

plt.plot(x, y, label="Original")
# plt.plot(x, y_spline, label="Spline")
plt.plot(real_x, real_y, label="Real", linestyle="--")
plt.legend()
plt.grid()
plt.show()


