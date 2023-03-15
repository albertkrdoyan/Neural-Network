import numpy as np
from numba import njit, prange
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

    eps = 1e-15*(1 - np.sum(b))/np.sum(b)

    for _i, v in enumerate(output):
        if b[_i] == 1:
            output[_i] = (np.exp(v) / sum_z) - eps
        else:
            output[_i] = 1e-15
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

def print_matrix_as_image(mx : np.ndarray):
    for line in mx:
        for cell in line:
            if cell != 0:
                print("▓", end='')
            else:
                print(" ", end='')
        print()

@njit(parallel=True)
def neuralMultiplicationMega(result, vect, mx: np.ndarray):
    #for i in range(len(result)):
    for i in prange(result.shape[0]):
        result[i] = 0
        for j in range(len(vect)):
            result[i] += vect[j] * mx[j][i]
        result[i] += mx[len(mx) - 1][i]
@njit(parallel=True)
def action_by_number_jit(result: np.ndarray, mx: np.ndarray, number: float, action):
    for row in prange(mx.shape[0]):
        for col in range(len(mx[0])):
            if action == "mul":
                result[row][col] = mx[row][col] * number
            elif action == "add":
                result[row][col] = mx[row][col] + number
            elif action == "sub":
                result[row][col] = mx[row][col] - number
            elif action == "div":
                result[row][col] = mx[row][col] / number
@njit(parallel=True)
def action_matrices_jit(result: np.ndarray, mx1: np.ndarray, mx2: np.ndarray, action):
    for row in prange(mx1.shape[0]):
        for col in range(len(mx1[0])):
            if action == "add":
                result[row][col] = mx1[row][col] + mx2[row][col]
            elif action == "sub":
                result[row][col] = mx1[row][col] - mx2[row][col]
            elif action == "mul":
                result[row][col] = mx1[row][col] * mx2[row][col]
            elif action == "div":
                result[row][col] = mx1[row][col] / mx2[row][col]
@njit#(parallel=True)
def exp_matrices_jit(result: np.ndarray, mx1: np.ndarray, exponent: float):
    for row in range(mx1.shape[0]):
        for col in range(len(mx1[0])):
            result[row][col] = mx1[row][col] ** exponent
