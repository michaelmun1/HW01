

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


#q2 ------------------------

x, y = np.loadtxt(r"C:\Users\micha\Downloads\HW01_data.txt", skiprows=1, unpack=True)

print(x)
print(y)


# plt.scatter(x, y)
# plt.xlabel("x")
# plt.ylabel("y")


x_hires = np.linspace(x.min(), x.max(), 10 * len(x))

# piecewise interpolation for more y points
y_hires = np.zeros_like(x_hires)

for j, xq in enumerate(x_hires):

    # wq interval check
    i = np.searchsorted(x, xq) - 1

    # I had to limit it to the valid range so endpoints don't break
    if i < 0:
        i = 0
    if i > len(x) - 2:
        i = len(x) - 2

    x0, x1 = x[i], x[i+1]
    y0, y1 = y[i], y[i+1]

    # linear interpolation formula
    y_hires[j] = y0 + (y1 - y0) * (xq - x0) / (x1 - x0)




#part b - cubic spline interpolation (off-the-shelf)
cs = CubicSpline(x, y, bc_type="natural")   # "natural" is a common default
y_spline = cs(x_hires)



# ooriginal + interpolated + cubic spline
plt.figure()
plt.scatter(x, y, label="original")
plt.scatter(x_hires, y_hires, s=10, label="linear interp points")   # 
plt.plot(x_hires, y_spline, label="cubic spline")                   
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Q2 Plot")
plt.show()

























