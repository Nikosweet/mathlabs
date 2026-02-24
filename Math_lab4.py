class functions:
    @staticmethod
    def f(x: float) -> float:
        return 4 * x**2

    @classmethod
    def first_derivative(cls, x: float = 0, h: float = 10**(-11)) -> float:
        result = (cls.f(x+h) - cls.f(x)) / (x+h - x)
        return result


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

NumberOne.first_number()
