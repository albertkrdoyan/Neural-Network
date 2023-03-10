import numpy as np
from math import exp
from numba import njit

@njit
def SoftMax(output : np.ndarray):
    if abs(np.max(output)) < 350 and abs(np.min(output)) < 350:
        sum_z = 0.
        for i, v in enumerate(output):
            sum_z += np.exp(v)
        for i, v in enumerate(output):
            output[i] = np.exp(v) / sum_z
        return output

    m, b, rng, sum_z = np.max(output), np.zeros(output.shape), 350, 0.

    for _i, v in enumerate(output):
        if m - v < 2 * rng:
            output[_i] = output[_i] - m + rng
            b[_i] = 1
    for _i, v in enumerate(output):
        if b[_i] == 1:
            sum_z += np.exp(v)

    eps = 1e-25 * (len(b) - np.sum(b)) / np.sum(b)

    for _i, v in enumerate(output):
        if b[_i] == 1:
            output[_i] = (np.exp(v) / sum_z) - eps
        else:
            output[_i] = 1e-25
    return output

def print_matrix(matrix, pref : str = ''):
    for i, row in enumerate(matrix):
        if i == 0:
            print('{}['.format(pref), end='')
            print('[', end='')
        else:
            print(' ', end='')
            print('\t [', end='')

        for c in range(len(row)):
            print(matrix[i][c], end='')
            if c != len(row) - 1:
                print(', ', end='')
        print(']', end='')
        if i != len(matrix) - 1:
            print(', ')
        else:
            print(']')


def create_matrix(col_count: int, row_count: int):
    return np.zeros((col_count, row_count), dtype='float64')


def copy_matrix_params(matrix):
    return create_matrix(len(matrix[0]), len(matrix))


@njit
def neuralMultiplicationMega(result, vect, mx: np.ndarray):
    for i in range(len(result)):
        result[i] = 0
        for j in range(len(vect)):
            result[i] += vect[j] * mx[j][i]
        result[i] += mx[len(mx) - 1][i]


@njit(debug=True)
def action_by_number_jit(result: np.ndarray, mx: np.ndarray, number: float, action):
    for row in range(len(mx)):
        for col in range(len(mx[0])):
            if action == "mul":
                result[row][col] = mx[row][col] * number
            elif action == "add":
                result[row][col] = mx[row][col] + number
            elif action == "sub":
                result[row][col] = mx[row][col] - number
            elif action == "div":
                result[row][col] = mx[row][col] / number


@njit
def action_matrices_jit(result: np.ndarray, mx1: np.ndarray, mx2: np.ndarray, action):
    for row in range(len(mx1)):
        for col in range(len(mx1[0])):
            if action == "add":
                result[row][col] = mx1[row][col] + mx2[row][col]
            elif action == "sub":
                result[row][col] = mx1[row][col] - mx2[row][col]
            elif action == "mul":
                result[row][col] = mx1[row][col] * mx2[row][col]
            elif action == "div":
                result[row][col] = mx1[row][col] / mx2[row][col]


@njit
def exp_matrices_jit(result: np.ndarray, mx1: np.ndarray, exponent: float):
    for row in range(len(mx1)):
        for col in range(len(mx1[0])):
            result[row][col] = mx1[row][col] ** exponent
