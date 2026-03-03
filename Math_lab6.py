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
    def Eiler(x0: float, y0: float, xn: float, h: float = 1e-5, *, func: Callable = Functions.f1):
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
    def Runge_Kutte_Merson(x0: float, y0: float, xn: float, h: float = 1e-5, e: float = 1e-120, *, func: Callable = Functions.f1):
        if not h:
            raise ValueError('incorrect value of h')
        if h > 0 and x0 > xn:
            h = -h

        min_h = 1e-240
        x = []
        y = []

        while (h > 0 and x0 < xn) or (h < 0 and x0 > xn):

            if abs(h) < min_h:
                print(f"Warning: Step too small ({h}), stopping at x={x0}")
                return y0

            k1 = h * func(x0, y0)
            k2 = h * func(x0 + h/3, y0 + k1/3)
            k3 = h * func(x0 + h/3, y0 + k1/6 + k2/6)
            k4 = h * func(x0 + h/2, y0 + k1/8 + 3*k3/8)
            k5 = h * func(x0 + h, y0 + k1/2 - 3*k3/2 + 2 * k4)

            y_accur = y0 + k1/6 + 2*k4/3 + k5 / 6
            y_draft = y0 + k1/2 - 3 * k3 / 2 + 2 * k4

            R = 0.2 * abs(y_accur - y_draft)
            if R > e: h /= 2

            elif R < e/64:
                y.append(y0)
                x.append(x0)
                y0 = y_accur
                x0 += h
                h *= 2

            else:
                y.append(y0)
                x.append(x0)
                y0 = y_accur
                x0 += h

        return (y0, x, y)

    @staticmethod
    def Eiler_Adams(x_prelast: float, y_prelast: float, xn: float, h: float = 1e-5, e: float = 1e-120, func: Callable = Functions.f1):

        if not h:
            raise ValueError('incorrect value of h')
        if h > 0 and x_prelast > xn:
            h = -h

        x = [x_prelast]
        y = [y_prelast]

        fx = func(x_prelast, y_prelast)

        x_last = x_prelast +h
        y_last = y_prelast + h * fx

        x.append(x_last)
        y.append(y_last)

        while (h > 0 and x_last < xn) or (h < 0 and x_last > xn):
            temp = y_last
            y_last += h/2 * (3 * func(x_last, y_last) - func(x_prelast, y_prelast))
            y_prelast = temp
            y.append(y_last)

            x_prelast = x_last
            x_last += h
            x.append(x_last)


        return (y_last, x, y)






    @classmethod
    def draw(cls, x0: float, y0: float, xn: float, h: float = 1e-5, *, func: Callable = Functions.f1, method: Callable = Eiler, name: str = 'Function'):
        y0, x, y = method(x0, y0, xn, h, func=func)
        plt.plot(x, y, 'b-', linewidth=2, label='function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Функция методом {name}а')
        plt.legend()
        plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
        plt.clf()

        return y0



#print(Solutions.Eiler_recursive(0, 0, 10)) #Велика вероятность переполнения стека вызовов
#print(Solutions.Eiler(0, 0, 10)[0])
#print(Solutions.Runge_Kutte_Merson(0, 0, 10))
#print(Solutions.Eiler_Adams(0, 0, 10))

print(Solutions.draw(0, 0, 10, method=Solutions.Eiler, name='Эйлер'))
print(Solutions.draw(0, 0, 10, method=Solutions.Runge_Kutte_Merson, name = 'Рунге-Кутте-Мерсон'))
print(Solutions.draw(0, 0, 10, method=Solutions.Eiler_Adams, name = 'Эйлер-Адамс'))