import numpy as np
import matplotlib.pyplot as plt
import os
# значения по умолчанию
x = np.array([-2, 0, 2, 3, 4], dtype=float)
y = np.array([18, 12, 7, -1, 0], dtype=float)

filename = "data.txt"

if os.path.exists(filename) and os.path.getsize(filename) > 0:
    try:
        data = np.loadtxt(filename)
        # ожидаем 2 строки: первая x, вторая y
        if data.shape[0] >= 2:
            x = data[0]
            y = data[1]
    except Exception as e:
        print("Ошибка чтения файла, используются значения по умолчанию:", e)
print("x =", x)
print("y =", y)
# ПУНКТ 1
# МНК через numpy (как математический пакет)
coef_ls1 = np.polyfit(x, y, 1)
coef_ls2 = np.polyfit(x, y, 2)
coef_ls3 = np.polyfit(x, y, 3)
# Интерполяция (4 степень)
coef_interp = np.polyfit(x, y, 4)
print("=== ПУНКТ 1 ===")
print("МНК 1 степени:", coef_ls1)
print("МНК 2 степени:", coef_ls2)
print("МНК 3 степени:", coef_ls3)
print("Интерполяционный многочлен:", coef_interp)
# Построение
x_plot = np.arange(min(x)-1, max(x)+1, 0.01)
plt.figure(figsize=(9,6))
plt.plot(x_plot, np.polyval(coef_ls1, x_plot), label="МНК 1")
plt.plot(x_plot, np.polyval(coef_ls2, x_plot), label="МНК 2")
plt.plot(x_plot, np.polyval(coef_ls3, x_plot), label="МНК 3")
plt.plot(x_plot, np.polyval(coef_interp, x_plot), label="Интерполяция 4")
plt.scatter(x, y)
plt.grid()
plt.legend()
plt.title("Пункт 1 — расчёты")
plt.show(block=False)
# ПУНКТ 2
def gauss(A, b):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    for i in range(n):
        max_row = np.argmax(abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        pivot = A[i, i]
        A[i] /= pivot
        b[i] /= pivot
        for j in range(i + 1, n):
            factor = A[j, i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]
    x_res = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_res[i] = b[i] - np.dot(A[i, i + 1:], x_res[i + 1:])
    return x_res
# МНК
def least_squares(x, y, degree):
    A = np.zeros((degree + 1, degree + 1))
    b = np.zeros(degree + 1)
    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i, j] = np.sum(x ** (i + j))
        b[i] = np.sum(y * x ** i)
    return gauss(A, b)
# ЛАГРАНЖ
def lagrange(x_nodes, y_nodes, x_val):
    n = len(x_nodes)
    result = 0
    for i in range(n):
        L = 1
        for j in range(n):
            if i != j:
                L *= (x_val - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * L
    return result

# НЬЮТОН
def divided_differences(x, y):
    n = len(x)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (x[j:n] - x[0:n - j])
    return coef

def newton(x_nodes, coef, x_val):
    n = len(coef)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x_val - x_nodes[i]) + coef[i]
    return result
# ВЫЧИСЛЕНИЯ
coef_ls1 = least_squares(x, y, 1)
coef_ls2 = least_squares(x, y, 2)
coef_ls3 = least_squares(x, y, 3)
coef_interp = least_squares(x, y, 4)
coef_newton = divided_differences(x, y)
print("=== ПУНКТ 2 ===")
print("МНК 1:", coef_ls1)
print("МНК 2:", coef_ls2)
print("МНК 3:", coef_ls3)
print("Интерполяция (4):", coef_interp)
print("\nЛагранж:")
for val in x:
    print(f"x={val}, y={lagrange(x, y, val)}")
print("\nНьютон:")
for val in x:
    print(f"x={val}, y={newton(x, coef_newton, val)}")
# ===== Графики =====
x_plot = np.arange(min(x)-1, max(x)+1, 0.01)
plt.figure(figsize=(9,6))
plt.plot(x_plot, np.polyval(coef_ls1[::-1], x_plot), label="МНК 1")
plt.plot(x_plot, np.polyval(coef_ls2[::-1], x_plot), label="МНК 2")
plt.plot(x_plot, np.polyval(coef_ls3[::-1], x_plot), label="МНК 3")
plt.plot(x_plot, np.polyval(coef_interp[::-1], x_plot), label="Интерполяция 4")
plt.plot(x_plot, [lagrange(x, y, val) for val in x_plot], '--', label="Лагранж")
plt.plot(x_plot, [newton(x, coef_newton, val) for val in x_plot], ':', label="Ньютон")
plt.scatter(x, y)
plt.grid()
plt.legend()
plt.title("Пункт 2 — программная реализация")
plt.show()