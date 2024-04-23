import numpy as np

x0 = 1
y0 = 1
y1_0 = 1
x_end = 2
h = 0.1

def f(x, y, y1):
    return ((x + 1) / x) * y1 + 2 * ((x - 1) / x) * y

def analytic_solution(x):
    return np.exp(2 * x) / (3 * np.exp(2)) + ((3 * x + 1) * np.exp(-x)) / (3 * np.exp(1))

def euler_method(x0, y0, y1_0, x_end, h):
    n = int((x_end - x0) / h) + 1
    x_values = np.linspace(x0, x_end, n)
    y_values = np.zeros_like(x_values)
    y1_values = np.zeros_like(x_values)

    y_values[0] = y0
    y1_values[0] = y1_0

    for i in range(1, n):
        y1_values[i] = y1_values[i - 1] + h * f(x_values[i - 1], y_values[i - 1], y1_values[i - 1])
        y_values[i] = y_values[i - 1] + h * y1_values[i]

    return x_values, y_values

def euler_cauchy_method(x0, y0, y1_0, x_end, h):
    n = int((x_end - x0) / h) + 1
    x_values = np.linspace(x0, x_end, n)
    y_values = np.zeros_like(x_values)
    y1_values = np.zeros_like(x_values)

    y_values[0] = y0
    y1_values[0] = y1_0

    for i in range(1, n):
        y1_half = y1_values[i - 1] + (h / 2) * f(x_values[i - 1], y_values[i - 1], y1_values[i - 1])
        y_half = y_values[i - 1] + (h / 2) * y1_values[i - 1]
        y1_values[i] = y1_values[i - 1] + h * f(x_values[i - 1] + (h / 2), y_half, y1_half)
        y_values[i] = y_values[i - 1] + h * y1_values[i]

    return x_values, y_values

def runge_kutta_method(x0, y0, y1_0, x_end, h):
    n = int((x_end - x0) / h) + 1
    x_values = np.linspace(x0, x_end, n)
    y_values = np.zeros_like(x_values)
    y1_values = np.zeros_like(x_values)

    y_values[0] = y0
    y1_values[0] = y1_0

    for i in range(1, n):
        k1 = h * f(x_values[i - 1], y_values[i - 1], y1_values[i - 1])
        k2 = h * f(x_values[i - 1] + h / 2, y_values[i - 1] + k1 / 2, y1_values[i - 1] + k1 / 2)
        k3 = h * f(x_values[i - 1] + h / 2, y_values[i - 1] + k2 / 2, y1_values[i - 1] + k2 / 2)
        k4 = h * f(x_values[i - 1] + h, y_values[i - 1] + k3, y1_values[i - 1] + k3)

        y1_values[i] = y1_values[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y_values[i] = y_values[i - 1] + h * y1_values[i]

    return x_values, y_values

x_euler, y_euler = euler_method(x0, y0, y1_0, x_end, h)
x_euler_cauchy, y_euler_cauchy = euler_cauchy_method(x0, y0, y1_0, x_end, h)
x_runge_kutta, y_runge_kutta = runge_kutta_method(x0, y0, y1_0, x_end, h)

x_analytic = np.linspace(x0, x_end, 100)
y_analytic = analytic_solution(x_analytic)

print("x\tТочне значення\t\tМ.Ейлера\t\tМ.Ейлера-Коші\tМ.Рунге-Кутта\tПохибка(М.Ейлера)\tПохибка(М.Ейлера-Коші)\tПохибка(М.Рунге-Кутта)")
x_values = np.arange(1, 2.1, 0.1)
for i, x in enumerate(x_values):
    analytic = analytic_solution(x)
    euler_error = abs(analytic - y_euler[i])
    euler_cauchy_error = abs(analytic - y_euler_cauchy[i])
    runge_kutta_error = abs(analytic - y_runge_kutta[i])
    print(f"{x:.1f}\t\t{analytic:.6f}\t\t{y_euler[i]:.6f}\t\t{y_euler_cauchy[i]:.6f}\t\t{y_runge_kutta[i]:.6f}\t\t{euler_error:.6f}\t\t\t{euler_cauchy_error:.6f}\t\t\t{runge_kutta_error:.6f}")
