

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



# original + interpolated + cubic spline
plt.figure()
plt.scatter(x, y, label="original")
plt.scatter(x_hires, y_hires, s=10, label="linear interp points")   # 
plt.plot(x_hires, y_spline, label="cubic spline")                   
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Q2 Plot")
plt.show()






#q3a ------------------------------------------------

def f(x):
    return np.sin((np.pi/2) * x) + x/2

# data from 0 to 10
x3 = np.arange(0, 11, 1, dtype=float)
y3 = f(x3)

# hires data
x3_hires = np.linspace(0, 10, 10 * len(x3))   # 110 points

#  linear interpolation --
y3_lin = np.zeros_like(x3_hires)

for j, xq in enumerate(x3_hires):
    i = np.searchsorted(x3, xq) - 1

    if i < 0:
        i = 0
    if i > len(x3) - 2:
        i = len(x3) - 2

    x0, x1 = x3[i], x3[i+1]
    y0, y1 = y3[i], y3[i+1]

    y3_lin[j] = y0 + (y1 - y0) * (xq - x0) / (x1 - x0)

#  cubic spline 
cs3 = CubicSpline(x3, y3, bc_type="natural")
y3_spline = cs3(x3_hires)

plt.figure()
plt.scatter(x3, y3, label="dataset (integers)")
plt.plot(x3_hires, y3_lin, label="linear interp")
plt.plot(x3_hires, y3_spline, label="cubic spline")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Q3a Plot")
plt.show()



# q3b ------------------------------------------------

# true function on the hires grid
y3_true = f(x3_hires)

# relative error arrays
rel_err_lin = np.zeros_like(y3_true)
rel_err_spline = np.zeros_like(y3_true)

# cant divide by zero at x = 0
mask = y3_true != 0

rel_err_lin[mask] = (y3_lin[mask] - y3_true[mask]) / y3_true[mask]
rel_err_spline[mask] = (y3_spline[mask] - y3_true[mask]) / y3_true[mask]

# in the case of undefined points, I use NaN so plotting is clean
rel_err_lin[~mask] = np.nan
rel_err_spline[~mask] = np.nan

# the relative error plot
plt.figure()
plt.plot(x3_hires, rel_err_lin, label="linear rel error")
plt.plot(x3_hires, rel_err_spline, label="cubic spline rel error")
plt.axhline(0)
plt.xlabel("x")
plt.ylabel("relative error")
plt.legend()
plt.title("Q3b - Relative Error")
plt.show()


















