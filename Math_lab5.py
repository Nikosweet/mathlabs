#Левых прямоугольников, средних прямоугольников, 5-го порядка аппроксимации
from typing import Callable

class Functions:
    @staticmethod
    def func(x):
        return (1 + x**0.5) / (1 + 4 * x + 3 * x**2)

    @classmethod
    def derivative(cls, x: list , y: list, h: float = 1e-6):
        if not x or not y or not h:
            raise ValueError('Ошибка в диференцировании: не заданы переменные')
        n = len(x)
        derivatives = [(-3*y[0] + 4*y[1] - y[2]) / (2*h)]
        for i in range(1, n-1):
            derivatives.append((y[i+1] - y[i-1])/(2*h))
        derivatives.append((3 * y[n - 1] - 4 * y[n - 2] + y[n - 3]) / (2 * h))
        return derivatives

    @classmethod
    def second_derivative(cls, x: list, y: list, h: float = 1e-6):
        if not x or not y or not h:
            raise ValueError('Ошибка в диференцировании: не заданы переменные')
        n = len(x)
        derivatives = []

        derivatives.append((2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / (h ** 2))

        for i in range(1, n - 1):
            derivatives.append(
                (y[i + 1] - 2 * y[i] + y[i - 1]) / (4 * h ** 2))

        derivatives.append((2 * y[n - 1] - 5 * y[n - 2] + 4 * y[n - 3] - y[n - 4]) / (h ** 2))

        return derivatives

    @classmethod
    def sixth_derivative(cls, x: list, y: list, h: float = 1e-6):
        if not x or not y or not h:
            raise ValueError('Ошибка в диференцировании: не заданы переменные')
        n = len(x)
        if n < 7:
            return [0]
        derivatives = []

        for i in range(3, n - 3):
            d6 = (y[i + 3] - 6 * y[i + 2] + 15 * y[i + 1] - 20 * y[i] + 15 * y[i - 1] - 6 * y[i - 2] + y[i - 3]) / (h ** 6)
            derivatives.append(d6)

        return derivatives

class Methods:
    @classmethod
    def left_rectangle(cls, x_start: float = 0.5, x_end: float = 4.5, h: float = 1e-3, with_r: bool = False, func: Callable = Functions.func) -> float:
        if not x_start or not x_end or not h or not with_r or not func:
            raise ValueError('Параметры left_rectangle заданы неверно!')
        if not callable(func):
            raise TypeError('func must be callable')
        total_sum: float = 0
        x = x_start
        if (with_r):
            x_list: list = []
            y_list: list = []

        while x < x_end:
            y = func(x)
            total_sum += y
            if (with_r):
                x_list.append(x)
                y_list.append(y)
            x += h

        if (with_r):
            derivatives = Functions.derivative(x_list, y_list, h)
            M1 = max(abs(d) for d in derivatives)
            R = (h / 2) * (x_end - x_start) * M1
            print(f'Погрешность R <= {abs(R)}')

        return h * total_sum

    @classmethod
    def average_rectangle(cls, x_start: float = 0.5, x_end: float = 4.5, h: float = 10e-3, with_r: bool = False, func: Callable = Functions.func) -> float:

        if not x_start or not x_end or not h or not with_r or not func:
            raise ValueError('Параметры left_rectangle заданы неверно!')
        if not callable(func):
            raise TypeError('func must be callable')

        total_sum = 0
        x = x_start
        if (with_r):
            x_list: list = []
            y_list: list = []
        while x < x_end:
            y = func(x)
            total_sum += func((x + x+h) / 2)
            if (with_r):
                x_list.append(x)
                y_list.append(y)
            x += h
        if (with_r):
            M2 = max(abs(b) for b in Functions.second_derivative(x_list, y_list, h))
            R = h**2 / 24 * M2 * (x_end - x_start)

            print(f'Погрешность R <= {abs(R)}')
        return h * total_sum

    @classmethod
    def sixth_order_accuracy(cls, x_start: float = 0.5, x_end: float = 4.5, h: float = 1e-3, with_r: bool = False, func: Callable = Functions.func) -> float:

        if not x_start or not x_end or not h or not with_r or not func:
            raise ValueError('Параметры left_rectangle заданы неверно!')
        if not callable(func):
            raise TypeError('func must be callable')

        n_segments = int((x_end - x_start) / h)

        if n_segments % 5 != 0:
            n_segments = 5 + (n_segments // 5) * 5
            h = (x_end - x_start) / n_segments
        total_sum = 0
        x = x_start

        if with_r:
            x_list = []
            y_list = []

        c0 = 5.0 / 288.0
        weights = [19, 75, 50, 50, 75, 19]

        while x < x_end:
            f0 = func(x)
            f1 = func(x + h)
            f2 = func(x + 2 * h)
            f3 = func(x + 3 * h)
            f4 = func(x + 4 * h)
            f5 = func(x + 5 * h)

            if with_r:
                x_list.extend([x, x + h, x + 2 * h, x + 3 * h, x + 4 * h, x + 5 * h])
                y_list.extend([f0, f1, f2, f3, f4, f5])

            block_sum = (weights[0] * f0 + weights[1] * f1 + weights[2] * f2 + weights[3] * f3 + weights[4] * f4 + weights[5] * f5)
            total_sum += block_sum

            x += 5 * h

        result = c0 * h * total_sum

        if with_r:
            sixth_derivs = Functions.sixth_derivative(x_list, y_list, h/2)
            M6 = max(abs(d) for d in sixth_derivs)
            R = (x_end - x_start) * (h ** 6) * M6 / 945
            print(f'Погрешность R <= {abs(R)}')

        return result

    @classmethod
    def find_h(cls, function: Callable, x_start: float = 0.5, x_end: float = 4.5, eps: float = 1e-3, p: int = 1):

        if p / int(p) != 1:
            print('Заданный порядок точности должен быть целочисленным! Продолжаем с округлением в меньшую сторону.')
            p = int(p)
        if p < 1:
            print('Заданный порядок точности должен быть натуральным числом! Продолжаем с порядком точности = 1')
            p = 1
        if eps <= 0:
            print('Ошибка в задании epsilon, продолжаем со значением по умолчанию!')
            eps = 1e-3
        if not x_start or not x_end or not function or not eps or not p:
            raise ValueError('Параметры left_rectangle заданы неверно!')
        if not callable(function):
            raise TypeError('func must be callable')

        h = (x_end - x_start) / 4

        while True:
            S1 = function(x_start, x_end, h)
            S2 = function(x_start, x_end, h/2)

            if (abs(S1-S2) / (2**p - 1) < eps):
                break
            else:
                h /= 2

        return function(x_start, x_end, h/2, True), h/2

print(f'S и h соответственно: {Methods.find_h(Methods.left_rectangle)}')
print(f'S и h соответственно: {Methods.find_h(Methods.average_rectangle, p=2)}')
print(f'S и h соответственно: {Methods.find_h(Methods.sixth_order_accuracy, p=6)}')