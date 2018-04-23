import random
import numpy as np

NUM_EXPRESSIONS = 10
MAX_NUMBER = 9


#TOKENS
PLUS = 11
EQUAL = 12
MUL = 13
MINUS = 14
EOS = 15
START_TOKEN = 16



vocab_size = 17


def get_data(num):
    input = []
    target = []
    for _ in range(num):
        x = random.randint(0, MAX_NUMBER)
        y = random.randint(0, MAX_NUMBER)

        r = random.randint(0,2)

        if r == 0:
            z = x + y
            input_s = str(x) + "+" + str(y) + "="
        elif r == 1:
            z = x * y
            input_s = str(x) + "*" + str(y) + "="
        elif r == 2:
            z = x - y
            input_s = str(x) + "-" + str(y) + "="

        if 10 > z >= 0:
            output_s = "0" + str(z)
        else:
            output_s = str(z)

        i, o = expression2array(input_s, output_s)

        input.append(i)
        target.append(o)

    return input, target


# return input, target
def expression2array(input_exp, output_exp):
    input_array = []
    for c in input_exp:
        if c == '+':
            input_array.append(PLUS)
        elif c == '*':
            input_array.append(MUL)
        elif c == '-':
            input_array.append(MINUS)
        elif c == '=':
            input_array.append(EQUAL)
        else:
            input_array.append(int(c))

    #input_array.append(EOS)

    output_array = []
    #output_array.append(START_TOKEN)

    for c in output_exp:
        if c == '-':
            output_array.append(MINUS)
        else:
            output_array.append(int(c))

    return input_array, output_array


def array2expression(array):
    array = np.array(array).reshape(1, -1)
    exp = ""
    for n in array[0]:
        if n == START_TOKEN or n == EOS :
            continue
        elif n == PLUS:
            exp = exp + "+"
        elif n == EQUAL:
            exp = exp + "="
        elif n == MUL:
            exp = exp + "*"
        elif n == MINUS:
            exp = exp + "-"
        else:
            exp = exp + str(n)

    return exp



# for _ in range(10):
#     print("______________")
#     x, y = get_data(1)
#     print(x)
#     print(y)
#     print(array2expression(x))
#     print(array2expression(y))



