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

@njit
def pre_back(data : tuple):
    loss_, netInfo_, output_, last_neuron_layer = data

    if loss_.value == "Categorical Cross Entropy":
        if netInfo_[-1][1].value == "SoftMax":
            for i in range(len(last_neuron_layer)):
                output_[i] = last_neuron_layer[i] - output_[i]
        else:
            for i in range(len(last_neuron_layer)):
                output_[i] = - output_[i] / last_neuron_layer[i]
    elif loss_.value == "Quadratic Loss":
        for i in range(len(last_neuron_layer)):
            output_[i] = 2 * (last_neuron_layer[i] - output_[i])

@njit(parallel=True)
def back_propagation_for_perceptron(data : tuple):
    activation, output_, outputs_, inputs_, weights_, gradient_ = data

    if activation == "ReLU":
        for i in prange(outputs_.shape[0]):
            if outputs_[i] == 0:
                output_[i] = 0
    elif activation == "Sigmoid":
        for i in prange(outputs_.shape[0]):
            output_[i] = output_[i] * outputs_[i] * (1 - outputs_[i])
    elif activation == "TanH":
        for i in prange(outputs_.shape[0]):
            output_[i] = output_[i] * (1 - outputs_[i]**2)

    derivative = np.zeros_like(inputs_, dtype='float64')
    for i in prange(derivative.shape[0]):
        for j in range(len(output_)):
            derivative[i] += output_[j] * weights_[i][j]

    bias_line = len(gradient_) - 1
    for i in prange(output_.shape[0]):
        gradient_[bias_line][i] += output_[i]
        for j, val_j in enumerate(inputs_):
            gradient_[j][i] += (output_[i] * val_j)

    jit_equal1d(inputs_, derivative)

@njit(parallel=True)
def jit_equal1d(arr1d_1 : np.ndarray, arr1d_2 : np.ndarray, new : bool = True):
    if new is True:
        arr1d_1 = np.zeros(arr1d_2.shape, dtype=arr1d_2.dtype)
    for ind in prange(arr1d_2.shape[0]):
        arr1d_1[ind] = arr1d_2[ind]


@njit(parallel=True)
def forward_propagation_for_perceptron(data : tuple):
    activation, outputs_, inputs_, weights_ = data
    neuralMultiplicationMega(outputs_, inputs_, weights_)

    if activation.value == "ReLU":
        for i in prange(outputs_.shape[0]):
            if outputs_[i] < 0:
                outputs_[i] = 0
    elif activation.value == "Sigmoid":
        for i in prange(outputs_.shape[0]):
            outputs_[i] = 1/(1 + np.exp(-outputs_[i]))
    elif activation.value == "SoftMax":
        SoftMax(outputs_)
    elif activation.value == "TanH":
        for i in prange(outputs_.shape[0]):
            outputs_[i] = np.tanh(-outputs_[i])

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
