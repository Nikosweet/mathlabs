from math import inf


class Matrix:
    @classmethod
    def input_matrix(cls):
        rows = int(input('Введите количество строк в матрицах: \n'))
        print('Введите строки матриц через пробел, начиная со значений А')
        matrix = []

        for i in range(rows):
            str = list(map(float, input().split(' ')))
            matrix.append(str)

        for row in matrix:
            if len(row) != rows+1:
                raise ValueError('Неправильно задана матрица!')
        if len(matrix) != rows:
            raise ValueError('Неверное количество строк в матрице!')

        return matrix, rows

    @classmethod
    def diagonal_dominance(cls, matrix: list = None, rows: int = None):
        if matrix == None or rows == None:
            (matrix, rows) = cls.input_matrix()

        obj = {}
        for row in matrix:
            result = row.pop()
            max_value = 0
            index = 0
            max_sum = 0
            for i in range(len(row)):
                if abs(row[i]) > max_value:
                    max_value = abs(row[i])
                    index = i
            for i in range(len(row)):
                if i != index:
                    max_sum += abs(row[i])
            if max_sum >= max_value:
                raise ValueError('Диагональное преобладание отсутствует!')
            if index in obj:
                raise ValueError('Диагональное преобладание отсутствует!')
            row.append(result)
            obj[index] = row
        new_matrix = []
        for i in range(len(obj)):
            new_matrix.append(obj[i])
        return [new_matrix, rows]


class Gauss(Matrix):
    @classmethod
    def straight_stroke_simple(cls, matrix: list = None, rows: int = None):
        if matrix is None or rows is None:
            (matrix, rows) = cls.input_matrix()


        n = len(matrix[0]) - 1

        for k in range(min(rows, n)):
            if abs(matrix[k][k]) < 1e-10:
                continue

            for i in range(k + 1, rows):
                coeff = matrix[i][k] / matrix[k][k]
                for j in range(k, n + 1):
                    matrix[i][j] -= coeff * matrix[k][j]
                    if abs(matrix[i][j]) < 1e-10:
                        matrix[i][j] = 0

        return [matrix, rows]


    @classmethod
    def straight_stroke(cls, matrix: list = None, rows: int = None):
        if matrix is None or rows is None:
            (matrix, rows) = cls.input_matrix()


        strA_len = len(matrix[0]) - 1
        k = 0

        for j in range(strA_len):
            max_el = 0
            max_str = k
            for i in range(k, rows):
                if max_el < abs(matrix[i][j]):
                    max_el = abs(matrix[i][j])
                    max_str = i

            if abs(max_el) < 1e-10:
                continue

            if max_str != k:
                str_temp = matrix[k]
                matrix[k] = matrix[max_str]
                matrix[max_str] = str_temp
            max_el = matrix[k][j]

            for i in range(k + 1, rows):
                el = matrix[i][j]
                coeff = el / max_el
                for m in range(strA_len + 1):
                    matrix[i][m] = matrix[i][m] - coeff * matrix[k][m]
                    if abs(matrix[i][m]) < 1e-10:
                        matrix[i][m] = 0
            k += 1
        return [matrix, rows]

    @classmethod
    def remove_null(cls, matrix: list = None, rows: int = None, *, simple: bool = False):
        if matrix == None or rows == None:
            if simple: (matrix, rows) = cls.straight_stroke_simple()
            else: (matrix, rows) = cls.straight_stroke()

        for i in range(rows - 1, -1, -1):
            count_not_null = 0
            for num in matrix[i]:
                if num != 0:
                    count_not_null += 1
            if count_not_null == 0:
                del matrix[i]
                raise ValueError('Данная система имеет бесконечно много решений!')
            if count_not_null == 1 and matrix[i][-1] != 0:
                raise ValueError("Данная система не имеет решений!")
        print(matrix)
        return [matrix, rows]

    @classmethod
    def back_stroke(cls, matrix: list = None, rows: int = None, *, simple: bool = False):
        if matrix == None or rows == None:
            if simple: (matrix, rows) = cls.remove_null(1)
            else: (matrix, rows) = cls.remove_null()
        if rows == len(matrix):
            print('Данная система имеет лишь одно решение!')
            answer = []
            k = rows
            for i in range(rows - 1, -1, -1):
                r = 0
                print(f'Строка: {matrix[i]}')
                for j in range(rows-1, k-1, -1): # Не запускается на первой итерации
                    matrix[i][-1] = matrix[i][-1] - (answer[r] * matrix[i][j])
                    r += 1
                else:
                    answer.append(matrix[i][-1] / matrix[i][k-1])
                    k -= 1
            return answer[::-1]



class Seidel(Matrix):
    @classmethod
    def check_data(cls, matrix: list = None, rows: int = None):
        if matrix == None or rows == None:
            (matrix, rows) = cls.input_matrix()
        for row in matrix:
            sum = 0
            central = 0
            for i in range(len(row)-1):
                if row[i] == matrix[i][i]: central = abs(row[i])
                else: sum += abs(row[i])
        if (central <= sum): raise ValueError('Диагональное преобладание отсутствует!')


    @classmethod
    def matrix_norm1(cls, matrix: list = None, rows: int = None):
        if matrix == None or rows == None:
            (matrix, rows) = cls.input_matrix()
        max_sum = 0
        for j in range(rows):
            sum = 0

            for i in range(rows):
                sum += abs(matrix[i][j])
            if sum > max_sum: max_sum = sum

        return max_sum


    @classmethod
    def matrix_normk(cls, matrix: list = None, rows: int = None):
        if matrix == None or rows == None:
            (matrix, rows) = cls.input_matrix()
        sum = 0

        for i in range(rows):
            for j in range(rows):
                sum += matrix[i][j]**2

        return sum**0.5


    @classmethod
    def method(cls, matrix: list = None, rows: int = None, *,
            diagonal_dominance: bool = True,
            condition_check: bool = True,
            good_ending_condition: bool = True):
        if matrix == None or rows == None:
            if diagonal_dominance: (matrix, rows) = cls.diagonal_dominance()
            else: (matrix, rows) = cls.input_matrix()


        e = 10e-10000
        if good_ending_condition:
            condition = (1 - cls.matrix_norm1(matrix, rows)) / cls.matrix_normk(matrix, rows) * e
        else: condition = e


        answers = [0 for row in matrix]
        last_step = inf
        if condition_check:
            cls.check_data(matrix, rows)
            print('Проверка сходимости выполнена успешно!')
        while (abs(last_step) > abs(condition)):
            max_change = 0
            print(last_step, condition)
            for i in range(rows):
                prev_answer = answers[i]
                answer = matrix[i][-1]
                for j in range(len(matrix[i]) - 1):
                    if i != j:
                        answer = answer - matrix[i][j] * answers[j]
                    else:
                        coeff = matrix[i][j]
                answers[i] = answer / coeff
                change = abs(answers[i] - prev_answer)
                if change > max_change:
                    max_change = change
            last_step = max_change

        return answers


class Thomas(Gauss):
    @classmethod
    def is_tridiagonal(cls, matrix: list = None, rows: int = None):
        if matrix == None or rows == None:
            (matrix, rows) = cls.input_matrix()

        for i in range(rows):
            if matrix[i][i] == 0:
                raise ValueError('Diagonal element equals 0!')
            for j in range(rows):
                if abs(i-j) > 1 and matrix[i][j] != 0:
                    raise ValueError('That is not tridiagonal!')


    @classmethod
    def method(cls, matrix: list = None, rows: int = None):
        if matrix == None or rows == None:
            (matrix, rows) = cls.input_matrix()

        (matrix, rows) = cls.diagonal_dominance(matrix, rows)

        cls.is_tridiagonal(matrix, rows)

        answers = [0 for _ in range(rows)]
        for i in range(1, rows):
            temp = matrix[i][i - 1] / matrix[i - 1][i - 1]
            matrix[i][i] -= temp * matrix[i - 1][i]
            matrix[i][-1] -= temp * matrix[i - 1][-1]

        answers[rows - 1] = matrix[rows - 1][-1] / matrix[rows - 1][rows - 1]

        for i in range(rows - 2, -1, -1):
            answers[i] = (matrix[i][-1] - matrix[i][i + 1] * answers[i + 1]) / matrix[i][i]
        return answers


#print('Метод Гаусса без выбора главного элемента\n', asyncio.run(Gauss.back_stroke(simple=True)))
#print('Метод Гаусса\n', asyncio.run(Gauss.back_stroke()))
#print('Метод Зейделя с перестановкой матрицы\n', asyncio.run(Seidel.method(condition_check=False)))
#print('Метод Зейделя с проверкой условия и перестановкой матрицы\n', asyncio.run(Seidel.method()))
#print('Метод Зейделя c проверкой условия и ошибкой, т.к. оно не выполняется\n', asyncio.run(Seidel.method(diagonal_dominance=False)))
#print('Метод Зейделя\n', asyncio.run(Seidel.method(condition_check=False, diagonal_dominance=False)))
#print('Метод Томаса\n', asyncio.run(Thomas.method()))

#Для гаусса:
'''
2 1 4 16
3 2 1 10
1 3 3 16


3 5 1 12
1,799999 3 7 13,599998
8 1 1 18
'''

'''
0 -1 2 -4 4
1 0 2 -2 6
4 2 8 1 -2
1 1 2 -1 8
'''

#Для Зейделя (необходимо диагональное преобладание)
'''
4 1 1 7
1 5 2 10
2 3 8 20
'''

'''
5 30 6 53 
-3 4 20 61
10 2 1 15
'''

#Для Томаса (Прогонки)
'''
4 -1 0 0 0 7
-1 4 -1 0 0 5
0 -1 4 -1 0 3
0 0 -1 4 -1 1
0 0 0 -1 4 3
'''
#Примерные ответы: 2.301269098185003, 2.2051184778101742, 1.5192259024479426, 0.8717930463608354, 0.9679482615902089