#Эйлера
#Рунге-Кутты-Мерсона
#Исправленный Эйлера Адамса 2-го порядка

import matplotlib.pyplot as plt
from math import sin
from typing import Callable


class Functions:
    @staticmethod
    def f1(x, y):
        return sin(x) - y + 10


class Solutions(Functions):
    @classmethod
    def Eiler_recursive(cls, x0: float, y0: float, xn: float, h: float = 1e-1, func: Callable = Functions.f1) -> float:
        if (x0 == xn or (x0 > xn and h > 0) or (x0 < xn and h < 0)):
            return y0
        if not h:
            raise ValueError('Incorrect value of h')

        fx = func(x0, y0)
        x_new = x0 + h
        y_new = y0 + h*fx

        return cls.Eiler_recursive(x_new, y_new, xn, h, func)

    @staticmethod
    def Eiler(x0: float, y0: float, xn: float, h: float = 1e-6, *, func: Callable = Functions.f1):
        if not h:
            raise ValueError('incorrect value of h')

        x = []
        y = []

        while (x0 < xn and h > 0):
            fx = func(x0, y0)
            x.append(x0)
            y.append(y0)
            x0 += h
            y0 += h*fx

        while (x0 > xn and h < 0):
            x.append(x0)
            y.append(y)
            fx = func(x0, y0)
            x0 += h
            y0 += h*fx

        return (y0, x, y)

    @staticmethod
    def Runge_Kutte_Merson(x0: float, y0: float, xn: float, h: float = 1e-3, e: float = 1e-21,
                           *, func: Callable = Functions.f1):
        if not h:
            raise ValueError('incorrect value of h')
        if h > 0 and x0 > xn:
            h = -h

        min_h = 1e-100
        max_h = abs(xn - x0)
        x = [x0]
        y = [y0]

        while (h > 0 and x0 < xn) or (h < 0 and x0 > xn):
            if abs(h) < min_h:
                print(f"Warning: Step too small ({h}), stopping at x={x0}")
                break

            if (h > 0 and x0 + h > xn) or (h < 0 and x0 + h < xn):
                h = xn - x0

            x_current = x0
            y_current = y0
            h_current = h

            k1 = h * func(x0, y0)
            k2 = h * func(x0 + h / 3, y0 + k1 / 3)
            k3 = h * func(x0 + h / 3, y0 + k1 / 6 + k2 / 6)
            k4 = h * func(x0 + h / 2, y0 + k1 / 8 + 3 * k3 / 8)
            k5 = h * func(x0 + h, y0 + k1 / 2 - 3 * k3 / 2 + 2 * k4)

            y_new = y0 + (k1 + 4 * k4 + k5) / 6

            error = abs(2 * k1 - 9 * k3 + 8 * k4 - k5) / 30

            if error > e:
                h = h_current / 2
                x0 = x_current
                y0 = y_current
                continue

            x0 = x_current + h_current
            y0 = y_new
            x.append(x0)
            y.append(y0)

            if error < e / 10:
                h = min(h_current * 2, max_h)
            elif error < e / 50:
                h = min(h_current * 1.5, max_h)

        return y0, x, y

    @staticmethod
    def Eiler_Adams(x_prelast: float, y_prelast: float, xn: float, h: float = 1e-14, e: float = 1e-14,
                    func: Callable = Functions.f1):
        if not h:
            raise ValueError('incorrect value of h')
        if h > 0 and x_prelast > xn:
            h = -h

        x = [x_prelast]
        y = [y_prelast]

        k1 = h * func(x_prelast, y_prelast)
        k2 = h * func(x_prelast + h / 2, y_prelast + k1 / 2)
        k3 = h * func(x_prelast + h / 2, y_prelast + k2 / 2)
        k4 = h * func(x_prelast + h, y_prelast + k3)

        x_last = x_prelast + h
        y_last = y_prelast + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        x.append(x_last)
        y.append(y_last)

        while (h > 0 and x_last < xn) or (h < 0 and x_last > xn):
            current_h = h
            current_x_prelast = x_prelast
            current_y_prelast = y_prelast
            current_x_last = x_last
            current_y_last = y_last

            f_last = func(current_x_last, current_y_last)
            f_prelast = func(current_x_prelast, current_y_prelast)
            y_predict = current_y_last + current_h / 2 * (3 * f_last - f_prelast)

            f_predict = func(current_x_last + current_h, y_predict)
            y_correct = current_y_last + current_h / 2 * (f_last + f_predict)

            error = abs(y_correct - y_predict) / 3

            if error > e:
                h = current_h / 2
                x_last = current_x_prelast
                y_last = current_y_prelast
                continue

            x_prelast = current_x_last
            y_prelast = current_y_last
            x_last = current_x_last + current_h
            y_last = y_correct

            x.append(x_last)
            y.append(y_last)

            if error < e / 10:
                h = current_h * 1.1
            else:
                h = current_h

        return y_last, x, y




    @classmethod
    def draw(cls, x0: float, y0: float, xn: float, h: float = 1e-6, *, func: Callable = Functions.f1, method: Callable = Eiler, name: str = 'Function'):
        y0, x, y = method(x0, y0, xn, h, func=func)
        plt.plot(x, y, 'b-', linewidth=2, label='function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Функция методом {name}а')
        plt.legend()
        plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
        plt.clf()

        return y0



#print(Solutions.Eiler_recursive(0, 0, 10)) # Велика вероятность переполнения стека вызовов
print('Эйлер, h = 10e-4 ',Solutions.Eiler(0, 0, 10, 10e-4)[0])
print('Рунге-Кутты-Мерсона, h = 10e-15, e = 10e-15',Solutions.Runge_Kutte_Merson(0, 0, 10, 10e-15, 10e-15)[0])
print("Эйлера-Адамса, h = 10e-10, e = 10e-10",Solutions.Eiler_Adams(0, 0, 10, 10e-10, 10e-10)[0])

print('Эйлер, h = 10e-2 ',Solutions.Eiler(0, 0, 10, 10e-2)[0])
print('Рунге-Кутты-Мерсона, h = 10e-10, e = 10e-10',Solutions.Runge_Kutte_Merson(0, 0, 10, 10e-10, 10e-10)[0])
print("Эйлера-Адамса, h = 6e-10, e = 6e-10",Solutions.Eiler_Adams(0, 0, 10, 6e-10, 6e-10)[0])

print('Эйлер, h = 1 ',Solutions.Eiler(0, 0, 10, 1)[0])
print('Рунге-Кутты-Мерсона, h = 5e-10, e = 5e-10',Solutions.Runge_Kutte_Merson(0, 0, 10, 5e-10, 5e-10)[0])
print("Эйлера-Адамса, h = 3e-10, e = 3e-10",Solutions.Eiler_Adams(0, 0, 10, 3e-10, 3e-10)[0])

print('Эйлер, h = 10 ',Solutions.Eiler(0, 0, 10, 10)[0])
print('Рунге-Кутты-Мерсона, h = 2e-10, e = 2e-10',Solutions.Runge_Kutte_Merson(0, 0, 10, 2e-10, 2e-10)[0])
print("Эйлера-Адамса, h = 2e-10, e = 2e-10",Solutions.Eiler_Adams(0, 0, 10, 2e-10, 2e-10)[0])

print('Рунге-Кутты-Мерсона, h = 1e-10, e = 1e-10',Solutions.Runge_Kutte_Merson(0, 0, 10, 1e-10, 1e-10)[0])
print("Эйлера-Адамса, h = 1e-10, e = 1e-10",Solutions.Eiler_Adams(0, 0, 10, 1e-10, 1e-10)[0])
print('Рунге-Кутты-Мерсона, h = 1, e = 1',Solutions.Runge_Kutte_Merson(0, 0, 10, 1, 1)[0])
print("Эйлера-Адамса, h = 1, e = 1",Solutions.Eiler_Adams(0, 0, 10, 1, 1)[0])


print('Рунге-Кутты-Мерсона, h = 10, e = 10',Solutions.Runge_Kutte_Merson(0, 0, 10, 10, 10)[0])
print("Эйлера-Адамса, h = 10, e = 10",Solutions.Eiler_Adams(0, 0, 10, 10, 10)[0])
#print(Solutions.draw(0, 0, 10, method=Solutions.Eiler, name='Эйлер'))
#print(Solutions.draw(0, 0, 10, method=Solutions.Runge_Kutte_Merson, name = 'Рунге-Кутте-Мерсон'))
#print(Solutions.draw(0, 0, 10, method=Solutions.Eiler_Adams, name = 'Эйлер-Адамс'))