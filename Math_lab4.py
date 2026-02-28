from typing import Callable
from Math_lab2 import Thomas
import matplotlib.pyplot as plt
import numpy as np

class functions:
    @staticmethod
    def f(x: float) -> float:
        return 4 * x**2

    @classmethod
    def first_derivative(cls, x: float = 0, func: Callable = f, h: float = 10e-6) -> float:
        return (func(x+h) - func(x)) / (x+h - x)

    @classmethod
    def first_derivative_second(cls, x: float = 0, func: Callable = f, h: float = 10e-6) -> float:
        return (func(x+h) - func(x-h))/ (2*h)

    @classmethod
    def second_derivative(cls, x: float = 0, func: Callable = f, h: float = 10e-6) -> float:
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

    @staticmethod
    def __cubic_spline_function(f: float, b: float, c: float, d: float, dx: float):
        return f + b*dx + c * dx**2 + d * dx**3

    @classmethod
    def __get_cubic_spline_function(cls, x: list, f: list, b: list, c: list, d: list):

        def spline_function(x_val: float) -> float:
            for i in range(len(x) - 1):
                if x[i] <= x_val <= x[i+1]:
                    dx = x_val - x[i]
                    return cls.__cubic_spline_function(f[i], b[i], c[i], d[i], dx)
            return 0.0

        return spline_function


    @classmethod
    def draw_cubic_spline(cls, x: list = None, f: list = None, b: list = None, c: list = None, d: list = None, *, with_derivative: bool = True):
        if not(f and b and c and d and x):
            if not(f and x):
                (x, f, b, c, d) = cls.cubic_spline([-2, 0, 2, 3, 4], [18, 12, 7, -1, 0])
            (x, f, b, c, d) = cls.cubic_spline(x, f)

        for i in range(len(x)-1):
            x_seg = np.linspace(x[i], x[i+1], 150)
            dx = x_seg - x[i]
            y_seg = cls.__cubic_spline_function(f[i], b[i], c[i], d[i], dx)

            plt.plot(x_seg, y_seg, 'b-', linewidth=2, label='Сплайн' if i == 0 else '')

        cls.draw_derivatives(x, f, b, c, d)

        plt.plot(x, f, 'ro', markersize=8, label='Узлы')
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Кубический сплайн')
        plt.legend()
        plt.savefig('cubicspline.png', dpi=300, bbox_inches='tight')

    @classmethod
    def draw_derivatives(cls, x: list = None, f: list = None, b: list = None,
                         c: list = None, d: list = None,
                         h_deriv: float = 1e-5, plot_step: float = 0.01):
        """
        Построение графиков производных сплайна

        Args:
            x: узлы сплайна
            f: значения в узлах
            b, c, d: коэффициенты сплайна
            h_deriv: шаг для ЧИСЛЕННОГО ДИФФЕРЕНЦИРОВАНИЯ (1e-5)
            plot_step: шаг для построения ГРАФИКА (0.01 = 10⁻²)
        """
        # Получаем коэффициенты, если не переданы
        if not (b and c and d):
            if not (f and x):
                x = [-2, 0, 2, 3, 4]
                f = [18, 12, 7, -1, 0]
            result = cls.cubic_spline(x, f)
            if result:
                x, f, b, c, d = result
            else:
                print("Ошибка построения сплайна")
                return

        # Получаем функцию сплайна
        spline_function = cls.__get_cubic_spline_function(x, f, b, c, d)

        # Создаем точки для графика с шагом plot_step
        x_dense = np.arange(min(x), max(x) + plot_step, plot_step)
        print(f"Создано {len(x_dense)} точек для графика")

        # Вычисляем значения
        y_spline = [spline_function(xi) for xi in x_dense]

        # ИСПРАВЛЕНО: используем правильные методы с правильным h
        y_first = []
        y_second = []

        for xi in x_dense:
            # Для каждой точки вычисляем производные ЧИСЛЕННО
            y_first.append(cls.first_derivative_second(xi, spline_function, h_deriv))
            y_second.append(cls.second_derivative(xi, spline_function, h_deriv))

        # Создаем фигуру
        fig, axes = plt.subplots(3, 1, figsize=(14, 16))

        # 1. Сам сплайн
        axes[0].plot(x_dense, y_spline, 'b-', linewidth=2, label='Сплайн S(x)')
        axes[0].plot(x, f, 'ro', markersize=8, label='Узлы', zorder=5)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=12)
        axes[0].set_ylabel('S(x)', fontsize=12)
        axes[0].set_title('Кубический сплайн', fontsize=14)

        # 2. Первая производная
        axes[1].plot(x_dense, y_first, 'r-', linewidth=2, label="S'(x) - численно")
        # Значения в узлах
        y_first_nodes = [cls.first_derivative_second(xi, spline_function, h_deriv) for xi in x]
        axes[1].plot(x, y_first_nodes, 'ko', markersize=6, label='Значения в узлах')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=12)
        axes[1].set_ylabel("S'(x)", fontsize=12)
        axes[1].set_title("Первая производная (численное дифференцирование)", fontsize=14)

        # 3. Вторая производная
        axes[2].plot(x_dense, y_second, 'g-', linewidth=2, label='S"(x) - численно')
        # Значения в узлах
        y_second_nodes = [cls.second_derivative(xi, spline_function, h_deriv) for xi in x]
        axes[2].plot(x, y_second_nodes, 'ko', markersize=6, label='Значения в узлах')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=12)
        axes[2].set_xlabel('x', fontsize=12)
        axes[2].set_ylabel('S"(x)', fontsize=12)
        axes[2].set_title("Вторая производная (численное дифференцирование)", fontsize=14)

        plt.tight_layout()
        plt.savefig('derivatives.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

CubicSpline.draw_derivatives([-2, 0, 2, 3, 4], [18, 12, 7, -1, 0])


