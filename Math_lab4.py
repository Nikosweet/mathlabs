from typing import Callable
from Math_lab2 import Thomas
import matplotlib.pyplot as plt
import numpy as np

class functions:
    @staticmethod
    def f(x: float) -> float:
        return 4 * x**2

    @classmethod
    def first_derivative(cls, x: float = 0, func: Callable = f, h: float = 10**(-11)) -> float:
        return (func(x+h) - func(x)) / (x+h - x)

    @classmethod
    def first_derivative_second(cls, x: float = 0, func: Callable = f, h: float = 10**(-11)) -> float:
        return (func(x+h) - func(x-h))/ (2*h)

    @classmethod
    def second_derivative(cls, x: float = 0, func: Callable = f, h: float = 10**(-11)) -> float:
        return (func(x+ 2 * h) - 2 * func(x) + func(x-2*h)) / (4 * h**2)

class NumberOne:
    @staticmethod
    def first_number() -> bool:
        try:
            print(functions.first_derivative(10**(-2)))
            print(functions.first_derivative(10**(-1)))
            print(functions.first_derivative(1))
            print(functions.first_derivative(10))
            print(functions.first_derivative(100))
            return True
        except Exception as e:
            return False

class CubicSpline(functions):
    @classmethod
    def cubic_spline(cls, x: list, f: list):
        try:
            h = []
            for i in range(1, len(x)):
                h.append(abs(x[i] - x[i-1]))
            n = len(h)-1
            A = [[0 for j in range(n)] for i in range(n)]
            B = [3 * ((f[i+2] - f[i+1]) / h[i+1] - ((f[i+1] - f[i]) / h[i])) for i in range(n)]
            A[0][0] = 2*(h[0]+h[1])
            A[0][1] = h[1]
            for i in range(1, n-1):
                A[i][i-1] = h[i]
                A[i][i] = 2*(h[i]+h[i+1])
                A[i][i+1] = h[i+1]
            else:
                A[-1][-1] = 2*(h[-2]+h[-1])
                A[-1][-2] = h[-2]
            for i in range(n): A[i].append(B[i])
            c = Thomas.method(A, n)
            c = [0, *c, 0]

            d = [(c[i+1] - c[i]) / (3 * h[i]) for i in range(n+1)]

            b = [(f[i+1] - f[i]) / h[i] - (2* c[i] + c[i+1])*h[i]/3 for i in range(n+1)]

            return (x, f, b, c, d)
        except Exception as e:
            print(e)

    @classmethod
    def draw_cubic_spline(cls, x: list = None, f: list = None, b: list = None, c: list = None, d: list = None):
        if not(f and b and c and d and x):
            if not(f and x):
                (x, f, b, c, d) = cls.cubic_spline([-2, 0, 2, 3, 4], [18, 12, 7, -1, 0])
            (x, f, b, c, d) = cls.cubic_spline(x, f)

        for i in range(len(x)-1):
            x_seg = np.linspace(x[i], x[i+1], 150)
            dx = x_seg - x[i]
            y_seg = f[i] + b[i]*dx + c[i] * dx**2 + d[i] * dx**3

            plt.plot(x_seg, y_seg, 'b-', linewidth=2, label='Сплайн' if i == 0 else '')

        plt.plot(x, f, 'ro', markersize=8, label='Узлы')
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Кубический сплайн')
        plt.legend()
        plt.savefig('cubicspline.png', dpi=300, bbox_inches='tight')

CubicSpline.draw_cubic_spline([-2, 0, 2, 3, 4], [18, 12, 7, -1, 0])


