import numpy as np
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

    eps = 1e-15*(1 - np.sum(b))/np.sum(b)

    for _i, v in enumerate(output):
        if b[_i] == 1:
            output[_i] = (np.exp(v) / sum_z) - eps
        else:
            output[_i] = 1e-15
    return output

@njit
def pre_back(data : tuple):
    loss_, netInfo_, output_, last_neuron_layer = data

    if loss_ == "Categorical Cross Entropy":
        if netInfo_[-1][1] == "SoftMax":
            for i in range(len(last_neuron_layer)):
                output_[i] = last_neuron_layer[i] - output_[i]
        else:
            for i in range(len(last_neuron_layer)):
                output_[i] = - output_[i] / last_neuron_layer[i]
    elif loss_ == "Quadratic Loss":
        for i in range(len(last_neuron_layer)):
            output_[i] = 2 * (last_neuron_layer[i] - output_[i])

@njit
def jit_equal1d(arr1d_1 : np.ndarray, arr1d_2 : np.ndarray):
    for ind in range(len(arr1d_2)):
        arr1d_1[ind] = arr1d_2[ind]
@njit
def forward_propagation_for_perceptron(data : tuple):
    activation, outputs_, inputs_, weights_ = data
    neuralMultiplicationMega(outputs_, inputs_, weights_)

    if activation == "ReLU":
        for i in range(len(outputs_)):
            if outputs_[i] < 0:
                outputs_[i] = 0
    elif activation == "Sigmoid":
        for i in range(len(outputs_)):
            outputs_[i] = 1/(1 + np.exp(-outputs_[i]))
    elif activation == "SoftMax":
        SoftMax(outputs_)
    elif activation == "TanH":
        for i in range(len(outputs_)):
            outputs_[i] = np.tanh(-outputs_[i])

def print_matrix_as_image(mx : np.ndarray):
    for line in mx:
        for cell in line:
            if cell != 0:
                print("▓", end='')
            else:
                print(" ", end='')
        print()

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