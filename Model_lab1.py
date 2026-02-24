from math import log10, log
import matplotlib.pyplot as plt
import numpy as np

class Serial:
    @classmethod
    def create_number(cls, length: int = 1024):
        arr = [0.12345]
        v = 3456
        total_small_series = 1
        total_big_series = 0

        curr_seria = 0
        curr_length = 1

        max_length_small_series = 1
        max_length_big_series = 0



        for i in range(1, length):
            temp = arr[i-1] * v - int(arr[i-1] * v)
            arr.append(round(temp, 10))
            if arr[i] >= 0.5:
                if curr_seria == 0:
                    total_big_series += 1
                    curr_seria = 1
                    if curr_length > max_length_small_series: max_length_small_series = curr_length
                    curr_length = 1
                else: curr_length += 1

            else:
                if curr_seria == 1:
                    total_small_series += 1
                    curr_seria = 0
                    if curr_length > max_length_big_series: max_length_big_series = curr_length
                    curr_length = 1
                else: curr_length += 1

        else:
            if curr_seria == 1 and curr_length > max_length_big_series: max_length_big_series = curr_length
            elif curr_seria == 0 and curr_length > max_length_small_series: max_length_small_series = curr_length
        cls.create_hystogramm(arr)
        return arr, total_small_series, total_big_series, max_length_big_series, max_length_small_series, length

    @classmethod
    def SerialMethod(cls, length: int = None, arr: list = None, total_small_series: int = None, total_big_series: int = None, max_length_big_series: int = None, max_length_small_series: int = None):
        if not arr or not total_small_series or not total_big_series or not max_length_big_series or not max_length_small_series:
            if not length:
                arr, total_small_series, total_big_series, max_length_big_series, max_length_small_series, length = cls.create_number()
            else:
                arr, total_small_series, total_big_series, max_length_big_series, max_length_small_series, length = cls.create_number(length)
        beta = 0.95
        R = 0.25 * (length - (1.63 * (length+1)**0.5))
        nmax = log10(-length/log(beta))/log10(2)-1
        print(f'''
length = {length}, 
Всего последовательностей < 0.5: {total_small_series},
Максимальная длина последовательностей < 0.5: {max_length_small_series},
Всего последовательностей >= 0.5: {total_big_series},
Максимальная длина последовательностей >= 0.5: {max_length_big_series}
nmax = {nmax}
R = {R}
''')
        if max_length_small_series > nmax or max_length_big_series > nmax:
            print('Гипотеза о случайности опровергается!')
            return 1
        if total_big_series <= R or total_small_series <= R:
            print('Гипотеза о случайности опровергается!')
            return 1
        print('Гипотеза о случайности не опровергается')
        return 0

    @classmethod
    def create_hystogramm(cls, data: list):
        data = np.array(data)
        plt.hist(data, bins='auto', edgecolor="black", linewidth=1.5)
        plt.xlabel('Значения')
        plt.ylabel('Частота')
        plt.title('Гистограмма')
        plt.savefig('histogram.png', dpi=300, bbox_inches='tight')


Serial.SerialMethod(16000)
